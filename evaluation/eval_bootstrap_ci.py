import numpy as np

def cal_avg_bootstrap_confidence_interval(x):
    x_avg = np.average(x)
    bootstrap_ci_result = bootstrap_ci(x)
    return np.round(x_avg, 4), np.round(bootstrap_ci_result[0], 4), np.round(bootstrap_ci_result[1], 4)

def bootstrap_ci(data, statistic=np.mean, alpha=0.05, num_samples=5000):
    n = len(data)
    rng = np.random.RandomState(47)
    samples = rng.choice(data, size=(num_samples, n), replace=True)
    stat = np.sort(statistic(samples, axis=1))
    lower = stat[int(alpha / 2 * num_samples)]
    upper = stat[int((1 - alpha / 2) * num_samples)]
    return lower, upper


if __name__ == '__main__':

    x = np.load('nnUnet_hd_metric.npy')
    print(x.reshape(-1))
    results = cal_avg_bootstrap_confidence_interval(x.reshape(-1))
