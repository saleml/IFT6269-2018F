import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import os
from inference import smoothing_conditional, multi_gaussian
from em import EMHMM

ROOT = 'hwk4data'
data_name = 'EMGaussian.'

MU = np.array([[-2.0344, 4.1726],
               [3.9799, 3.7735],
               [3.8007, -3.7972],
               [-3.0620, -3.5345]])

SIGMA = np.array([[[2.9044, 0.2066], [0.2066, 2.7562]],
                  [[0.2104, 0.2904], [0.2904, 12.2392]],
                  [[0.9213, 0.0574], [0.0574, 1.8660]],
                  [[6.2414, 6.0502], [6.0502, 6.1825]]])

pi = 1./4 * np.ones(4)

A = 1./6 * np.ones((4, 4)) + 1./3 * np.eye(4)


def sigmoid(z):
    return 1. / (1 + np.exp(- z))


def load_data(set_name='train', shuffle=False):
    with open(os.path.join(ROOT, data_name + set_name)) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [x.split() for x in content]
    X = np.array([[float(element) for element in x] for x in content])
    if set_name == 'train' and shuffle:
        np.random.shuffle(X)
    return X


def scatter_plot(X, title='', colors=None, ax=None, **kwargs):
    if ax is None:
        ax = plt
    ax.scatter(X[:, 0], X[:, 1], c=colors, **kwargs)
    ax.set_title(title)


def plot_datasets(return_data=False, colors=None, shuffle=False):
    rows, cols = 1, 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7))
    k = 0
    for set_name in ('train', 'test'):
        print(set_name)
        X = load_data(set_name, shuffle=shuffle)
        scatter_plot(X, title=set_name, colors=colors, ax=axes[k])
        k += 1
        if return_data:
            yield X


def plot_dataset(colors=None, set_name='train'):
    fig, ax = plt.subplots(figsize=(8, 7))
    k = 0
    X = load_data(set_name)
    scatter_plot(X, title=set_name, colors=colors, ax=ax)


def fake_parameters_inference(X):
    emission_density = multi_gaussian(MU, SIGMA)

    p, _, _ = smoothing_conditional(pi, X, A, emission_density)

    return p


def plot_hidden_state_probas(X, fake=True, model=None):
    K = 4
    if fake:
        p = fake_parameters_inference(X)
    else:
        assert model is not None
        p, _, _ = model.E_step(X)

    rows, cols = K, 1
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10), sharex=True)

    colors = 'ymbg'

    for k in range(K):
        axes[k].plot(range(100), p[:100, k], '{}o-'.format(colors[k]))
        axes[k].set_title('Probability of belong to class {}'.format(k + 1))

    axes[k].set_xlabel('Test set index')

    return p


def plot_most_likely_state(p):
    K = 4
    most_likely_state = np.argmax(p, axis=1)
    fig, ax = plt.subplots(figsize=(20, 3))
    ax.plot(range(100), most_likely_state[:100], 'ro-')
    ax.set_xlabel('Test set index')
    ax.set_title('Most likely state')


def train_hmm(X, X_test, K=4, var=1, d=2, plot=False):
    model = EMHMM(K=K, var=var, d=d, mu=MU, sigma=SIGMA, A=A, pi=pi)
    tau1, tau2, train_log_likelihoods, test_log_likelihoods = model.fit(X, X_test)

    if plot:
        fig, ax = plt.subplots(figsize=(20, 7))
        ax.plot(train_log_likelihoods, 'ro-', label='Average log likelihood on the train set')
        ax.plot(test_log_likelihoods, 'bo-', label='Average log likelihood on the test set')
        ax.set_xlabel('Step of the EM algorithm')
        ax.set_title('Average (over each data point, and not the whole chain :) ) log-likelihood ')
        ax.legend()

    hidden_states = model.viterbi_decoding(X)
    hidden_states_test = model.viterbi_decoding(X_test)

    return X, model, tau1.argmax(axis=1), tau2, train_log_likelihoods, test_log_likelihoods, hidden_states, hidden_states_test


def plot_hidden_states(hidden_states):
    fig, ax = plt.subplots(figsize=(20, 3))
    ax.plot(range(100), hidden_states[:100], 'ro-')
    ax.set_xlabel('Test set index')
    ax.set_title('Most likely state')


def plot_cov_ellipse(cov, pos, nstd=2.15, ax=None, **kwargs):
    """
    Copy-pasted/Modified from https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2.15 standard deviations. Which represents a 90% mass.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    ax.legend()
    return ellip


def plot_hmm(X, model, hidden_states):
    K = 4
    fig, ax = plt.subplots(figsize=(7, 7))

    means = model.mu
    covariances = model.sigma

    scatter_plot(X, title='EM for HMM',
                 colors=hidden_states, ax=ax)
    ax.plot(means[:, 0], means[:, 1], 'rx', mew=5, ms=10, label='Gaussian means')
    for k in range(K):
        plot_cov_ellipse(covariances[k], means[k], nstd=2.15, ax=ax, **{'fill': False})
    ax.legend()
    xyMIN = min(X[:, 0].min(), X[:, 1].min())
    xyMAX = max(X[:, 0].max(), X[:, 1].max())

    if xyMIN < 0:
        xyMIN *= 1.4
    else:
        xyMIN = 0
    if xyMAX > 0:
        xyMAX *= 1.4
    else:
        xyMAX = 0

    # Make the plot square so that spherical covariances are circles :)
    ax.set_xlim(xyMIN, xyMAX)
    ax.set_ylim(xyMIN, xyMAX)








