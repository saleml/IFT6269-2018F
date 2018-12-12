import numpy as np

def sigmoid(z):
    return 1./(1. + np.exp(-z))

def get_neighbors(i, j):
    right = (i, (j + 1) % 7)
    left = (i, (j - 1 + 7) % 7)
    up = ((i - 1 + 7) % 7, j)
    down = ((i + 1) % 7, j)
    return right, left, up, down

eta_st = 0.5
eta_s = [(-1) ** i for i in range(1,50)]
eta_s = np.reshape(eta_s, (7,7))


def new_sample(i, j, X):
    neighbors = get_neighbors(i, j)
    z = eta_s[i, j] + sum([eta_st * X[neighbor] for neighbor in neighbors])
    return np.random.binomial(1, sigmoid(z))

def epoch_update(X):
    new_X = np.copy(X)
    for i in range(7):
        for j in range(7):
            new_X[i, j] = new_sample(i, j, new_X)
    return new_X

def update_N_epochs(N, X):
    samples = [X]
    for _ in range(N):
        samples.append(epoch_update(samples[-1]))
    return samples


def mean_estimates(N, mixed_X):
    return np.mean(update_N_epochs(N, mixed_X), 0)


def mf_update(node, tau):
    return sigmoid(eta_s[node] + sum([eta_st * tau[neighbor] for neighbor in get_neighbors(*node)]))

def mean_field_epoch_update(tau):
    new_tau = np.copy(tau)
    for i in range(7):
        for j in range(7):
            new_tau[i, j] = mf_update((i, j), new_tau)
    return new_tau

def almost_kl(tau):
    value = np.sum(tau * np.log(tau) + (1 - tau) * np.log(1 - tau))
    for i in range(7):
        for j in range(7):
            node = (i, j)
            value -= eta_s[node] * tau[node] + eta_st * tau[node] * np.sum([tau[neighbor] for neighbor in get_neighbors(*node)])
    return value

def mean_field(eps=1e-3):
    tau = np.random.rand(7, 7)
    KLs = [almost_kl(tau)]
    while True:
        last_tau = tau
        tau = mean_field_epoch_update(last_tau)
        KLs.append(almost_kl(tau))
        d = np.mean(np.abs(tau - last_tau))
        if d < eps:
            break
    return tau, KLs
