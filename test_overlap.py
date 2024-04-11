import paddle
import argparse
import paddle.distributed as dist
from paddle.distributed import fleet
import paddle.nn as nn
from paddle.optimizer import AdamW
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.incubate.nn.layer.fused_dropout_add import FusedDropoutAdd
import paddle.tensor as tensor
import paddle.nn.functional as F


class Config:
    def __init__(self):
        self.vocab_size = 300000
        self.hidden_size = 1024
        self.seq_len = 1024
        self.batch_size = 1
        self.num_attention_heads = 8
        self.hidden_dropout_prob = 0
        self.attention_probs_dropout_prob = 0
        self.num_layers = 1


config = Config()


def gen_batch_data():
    global config
    input_ids = paddle.randint(low=0, high=config.vocab_size, shape=[config.batch_size, config.seq_len], dtype="int64")
    labels = paddle.randint(low=0, high=config.vocab_size, shape=[config.batch_size, config.seq_len], dtype="int64")
    loss_mask = paddle.ones(shape=[config.batch_size, config.seq_len], dtype="int64")
    return input_ids, labels, loss_mask


class SwiGLU(nn.Layer):
    def __init__(self):
        super().__init__()
        self.act = paddle.nn.Silu()

    def forward(self, x, gate):
        return self.act(x) * gate


class LayerNorm(nn.LayerNorm):
    def __init__(self, hidden_size, epsilon=1e-5):
        super().__init__(hidden_size, epsilon=epsilon)

    def forward(self, hidden_states):
        return super().forward(hidden_states)


class DecoderLayer(nn.Layer):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        global config

        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_dim = config.hidden_size // config.num_attention_heads

        multiple_of = 4 * config.num_attention_heads
        intermediate_size = multiple_of * ((int(8 * config.hidden_size // 3) + multiple_of - 1) // multiple_of)

        self.linear1 = nn.Linear(config.hidden_size, 2 * intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, config.hidden_size)

        self.activation = SwiGLU()

        self.norm1 = LayerNorm(config.hidden_size, epsilon=1e-5)
        self.norm2 = LayerNorm(config.hidden_size, epsilon=1e-5)

        self.fused_dropout_add1 = FusedDropoutAdd(config.hidden_dropout_prob, mode="upscale_in_train")
        self.fused_dropout_add2 = FusedDropoutAdd(config.attention_probs_dropout_prob, mode="upscale_in_train")

    def forward(self, hidden_states, mask):
        # attention
        tgt = hidden_states
        residual = tgt
        tgt = self.norm1(tgt)

        # get q, k, v
        query, key, value = tgt, tgt, tgt
        bsz, seq_len, _ = query.shape
        mix_layer = self.qkv_proj(query)
        mix_layer = paddle.reshape_(mix_layer, [bsz, seq_len, -1, 3 * self.head_dim])
        q, k, v = paddle.split(mix_layer, [self.head_dim, self.head_dim, self.head_dim], axis=-1)

        # none fa core attention
        perm = [0, 2, 1, 3]
        origin_dtype = q.dtype
        q = tensor.transpose(x=q, perm=perm)
        k = tensor.transpose(x=k, perm=perm)
        v = tensor.transpose(x=v, perm=perm)
        scale_qk_coeff = self.head_dim ** 0.5
        product = paddle.matmul(x=q.scale(1.0 / scale_qk_coeff), y=k, transpose_y=True)
        product = product.cast(paddle.float32)
        mask = mask.cast(paddle.float32)
        product = product + mask
        weights = F.softmax(product)
        weights = weights.cast(origin_dtype)
        out = paddle.matmul(weights, v)
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        tgt = tensor.reshape(x=out, shape=[0, 0, -1])

        # out proj of attention
        tgt = self.out_proj(tgt)
        tgt = self.fused_dropout_add1(tgt, residual)

        # ffn
        residual = tgt
        tgt = self.norm2(tgt)
        tgt_fc = self.linear1(tgt)
        shape = tgt_fc.shape[:-1]
        tgt_fc_ = tgt_fc.reshape(shape + [-1, 2])
        tgt_fc_shard0, tgt_fc_shard1 = paddle.chunk(tgt_fc_, 2, axis=-1)
        ffn_fc1, gate = tgt_fc_shard0.squeeze(-1), tgt_fc_shard1.squeeze(-1)
        tgt = self.activation(ffn_fc1, gate)
        tgt = self.linear2(tgt)
        tgt = self.fused_dropout_add2(tgt, residual)

        return tgt


class Model(nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        global config
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.decoder_layers = nn.LayerList()
        for i in range(config.num_layers):
            self.decoder_layers.append(DecoderLayer())

        self.norm = LayerNorm(config.hidden_size, epsilon=1e-5)

        self.out_linear = nn.Linear(config.hidden_size, config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, input_ids, labels, loss_mask):
        # embedding
        hidden_states = self.embedding(input_ids).astype(self.embedding.weight.dtype)

        # attn_mask
        seq_length = input_ids.shape[-1]
        attention_mask = paddle.tensor.tril(paddle.ones([seq_length, seq_length]))
        # [s, s] -> [1, 1, s, s]
        attention_mask = attention_mask[None, None]
        attention_mask.astype("bool")

        # decoder_layer
        for i, mod in enumerate(self.decoder_layers):
            hidden_states = mod(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)

        # out linear
        logits = self.out_linear(hidden_states)

        # criteria
        loss = self.loss_fn(logits, labels.unsqueeze(2))
        loss = paddle.sum(loss.reshape([-1]).cast(paddle.float32) * loss_mask.reshape([-1]).cast(paddle.float32))
        loss = loss / loss_mask.sum()
        return loss


def main():
    # parse arges
    parser = argparse.ArgumentParser()
    parser.add_argument('--overlap', type=str)
    args = parser.parse_args()
    overlap = (args.overlap == '1')
    print(f'overlap: {overlap}')

    # fleet init
    paddle.seed(1234)
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "sharding_degree": 2
    }
    strategy.hybrid_configs["sharding_configs"].tensor_fusion = True
    strategy.hybrid_configs["sharding_configs"].comm_overlap = overlap
    strategy.hybrid_configs["sharding_configs"].accumulate_steps = 1
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()

    # build model
    model = Model()
    optimizer = AdamW(learning_rate=3e-5, parameters=model.parameters(), weight_decay=0.01)

    # amp auto cast
    model = paddle.amp.decorate(
        models=model,
        level="O2",
        dtype="float16"
    )
    scaler = paddle.amp.GradScaler(init_loss_scaling=8192)

    # main grad
    mix_precision_utils.MixPrecisionScaler(scaler)
    mix_precision_utils.MixPrecisionLayer(model, dtype="float16")
    optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)

    # distributed model and optimizer
    optimizer = fleet.distributed_optimizer(optimizer)
    scaler = fleet.distributed_scaler(scaler)
    model = fleet.distributed_model(model)

    for i in range(50):
        input_ids, labels, loss_mask = gen_batch_data()
        with paddle.amp.auto_cast(
                enable=True,
                custom_white_list=["lookup_table", "lookup_table_v2", "flash_attn_npu", "matmul", "matmul_v2",
                                   "fused_gemm_epilogue"],
                custom_black_list=["reduce_sum", "c_softmax_with_cross_entropy", "elementwise_div", "sin", "cos"],
                level='O2',
                dtype='float16'
        ):
            loss = model(input_ids, labels, loss_mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad()

            # calc avg loss
            tr_loss = loss.detach()
            dist.all_reduce(
                tr_loss, dist.ReduceOp.SUM
            )
            tr_loss_scalar = tr_loss.item() / dist.get_world_size()
            tr_loss.zero_()
            print(f'step {i} loss {tr_loss_scalar}')


if __name__ == '__main__':
    main()