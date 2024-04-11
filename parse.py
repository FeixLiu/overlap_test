from matplotlib import pyplot as plt
import numpy as np


def parse_file(file_name):
    speeds = []
    losses = []
    idxs = []
    idx = 0
    with open(file_name) as file:
        for line in file:
            line = line.strip()
            if 'step' not in line:
                continue
            line = line.split(' ')
            loss = float(line[-1])
            losses.append(loss)
            idxs.append(idx)
            idx += 1
    return speeds, losses, idxs


baseline_loss = None
baseline_idx = None
baseline = None

start_idx = 0
end_idx = -1
plot_diff = True

if plot_diff:
    plt.figure(figsize=(14, 7))


file_names = [
    'tensor_fusion/workerlog.0',
    'tensor_fusion_0/workerlog.0',
    'overlap/workerlog.0',
    'overlap_0/workerlog.0',
    ]
max_diff_idx = 0
for file in file_names:
    speed, loss, idx = parse_file(file)
    if baseline_loss is None:
        baseline_loss = loss
        baseline_idx = idx
        baseline = file
    elif plot_diff:
        diff = []
        diff_idx = []
        for i in range(min(len(baseline_loss), len(loss))):
            diff.append(loss[i] - baseline_loss[i])
            diff_idx.append(i)
        max_diff_idx = max(max_diff_idx, len(diff_idx))
        plt.subplot(1, 2, 2)
        mean_diff = np.array(diff[start_idx:end_idx]).mean()
        print(f'mean {file} - {baseline}: {mean_diff}')
        plt.plot(diff_idx[start_idx:end_idx], diff[start_idx:end_idx], label=f'{file} - {baseline}')
        plt.legend()

    if plot_diff:
        plt.subplot(1, 2, 1)
    plt.plot(idx, loss, label=file)
    plt.legend()

plt.subplot(1, 2, 2)
zeros = [0 for i in range(max_diff_idx)]
diff_idx = [i for i in range(max_diff_idx)]
plt.plot(diff_idx[start_idx:end_idx], zeros[start_idx:end_idx], label='zero')

plt.legend()
plt.savefig('./plot.png')
