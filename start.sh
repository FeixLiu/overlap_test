source /usr/local/Ascend/ascend-toolkit/set_env.sh

export PYTHONPATH=$(dirname "$0")/..:$PYTHONPATH
export FLAGS_set_to_1d=False
export NVIDIA_TF32_OVERRIDE=0
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_SOCKET_IFNAME==xgbe0
export FLAGS_npu_storage_format=0

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINERS_NUM
unset PADDLE_TRAINER_ID
unset PADDLE_WORKERS_IP_PORT_LIST
unset PADDLE_TRAINERS
unset PADDLE_NUM_GRADIENT_SERVERS

export ACL_OP_DETERMINISTIC=true
export ACL_OPT_DETERMINISTIC=true
export HCCL_DETERMINISTIC=true
export npu_deterministic=true

rm -rf overlap
rm -rf overlap_0
rm -rf tensor_fusion
rm -rf tensor_fusion_0

rm -rf /root/.cache
rm -rf ./kernel_meta*
rm -rf *json

ps aux | grep test_overlap.py | grep -v grep | awk '{print $2}' | xargs kill -9

python -m paddle.distributed.launch \
    --log_dir overlap \
    --gpus "0,1" \
    test_overlap.py \
    --overlap 1

sleep 5

rm -rf /root/.cache
rm -rf ./kernel_meta*
rm -rf *json

ps aux | grep test_overlap.py | grep -v grep | awk '{print $2}' | xargs kill -9

python -m paddle.distributed.launch \
    --log_dir overlap_0 \
    --gpus "0,1" \
    test_overlap.py \
    --overlap 1

sleep 5

rm -rf /root/.cache
rm -rf ./kernel_meta*
rm -rf *json

ps aux | grep test_overlap.py | grep -v grep | awk '{print $2}' | xargs kill -9

python -m paddle.distributed.launch \
    --log_dir tensor_fusion \
    --gpus "0,1" \
    test_overlap.py \
    --overlap 0

sleep 5

rm -rf /root/.cache
rm -rf ./kernel_meta*
rm -rf *json

ps aux | grep test_overlap.py | grep -v grep | awk '{print $2}' | xargs kill -9

python -m paddle.distributed.launch \
    --log_dir tensor_fusion_0 \
    --gpus "0,1" \
    test_overlap.py \
    --overlap 0

python parse.py