import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import os
from kmeans import kmeans
from em import EMGMM

ROOT = 'hwk3data'
data_name = 'EMGaussian.'


def sigmoid(z):
    return 1. / (1 + np.exp(- z))


def load_data(set_name='train'):
    with open(os.path.join(ROOT, data_name + set_name)) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [x.split() for x in content]
    X = np.array([[float(element) for element in x] for x in content])
    return X


def scatter_plot(X, title='', colors=None, ax=None, **kwargs):
    if ax is None:
        ax = plt
    ax.scatter(X[:, 0], X[:, 1], c=colors, **kwargs)
    ax.set_title(title)


def plot_datasets():
    rows, cols = 1, 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7))
    k = 0
    for set_name in ('train', 'test'):
        X = load_data(set_name)
        scatter_plot(X, title=set_name, ax=axes[k])
        k += 1


def train_kmeans(K=4, std=1):
    X = load_data('train')
    model = kmeans(K=K, std=std)
    kmeans_objective_values = model.fit(X)
    assignment_array = model.cluster_assignments(X, False)
    return X, assignment_array, model.centroids, kmeans_objective_values


def plot_kmeans(K=4, std=1):
    X, assignment_array, centroids, kmeans_objective_values = train_kmeans(K, std)
    fig, ax = plt.subplots()
    scatter_plot(X, title='Clustering of the training data', colors=assignment_array, ax=ax)
    ax.plot(centroids[:, 0], centroids[:, 1], 'rx', mew=5, ms=10, label='centroids')
    ax.legend()


def analyze_initialization(K=4, std=1, N=1000, d=2):
    # Here d is the dimension of the data
    all_centroids = np.zeros((N, K, d))
    last_objectives = []
    for i in range(N):
        _, _, centroids, objective = train_kmeans(K, std)
        all_centroids[i] = centroids[centroids[:, 0].argsort()]
        last_objectives.append(objective[-1])

    return ((all_centroids.mean(axis=0), all_centroids.min(axis=0),
            all_centroids.max(axis=0), all_centroids.std(axis=0)),
            last_objectives, all_centroids)


def plot_initialization_analysis(K=4, std=1, N=1000, d=2, return_degenerate=False):
    centroids_info, objectives_info, all_centroids = analyze_initialization(K, std, N, d)
    mean_centroids, _, _, std_centroids = centroids_info

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    X = load_data('train')
    ax = axes[0]
    scatter_plot(X, title='Effects of initialization on Centroids', ax=ax)
    ax.plot(mean_centroids[:, 0], mean_centroids[:, 1], 'rx', mew=1, ms=5, label='mean of centroids')
    for k in range(K):
        # UNCOMMENT THE FOLLOWING LINES TO GET COORDINATE-WISE CONFIDENCE INTERVALS INSTEAD OF ELLIPSES
        # ax.plot(np.linspace(mean_centroids[k, 0] - 5 * std_centroids[k, 0],
        #                     mean_centroids[k, 0] + 5 * std_centroids[k, 0],
        #                     num=100),
        #         [mean_centroids[k, 1]] * 100,
        #         'k--')
        # ax.plot([mean_centroids[k, 0]] * 100,
        #         np.linspace(mean_centroids[k, 1] - 5 * std_centroids[k, 1],
        #                     mean_centroids[k, 1] + 5 * std_centroids[k, 1],
        #                     num=100), 'k--', label=('Confidence interval of the centroids' if k == 0 else None)
        #         )
        cov_centroids = np.cov(all_centroids[:, k, :].T)  # should be d x d
        plot_cov_ellipse(cov_centroids, mean_centroids[k], nstd=np.sqrt(10.597), ax=ax, **{'fill': False})
        # 10.597 corresponds to an ellipse containing 99.5% of the mass of the Gaussian
    ax.legend()
    axes[1].hist(objectives_info, 50)
    axes[1].set_title('Distribution of the final value of the objective function ({} runs)'.format(N))
    if return_degenerate:
        std_objectives = np.std(objectives_info)
        if std_objectives < 100:
            return None
        argmax_objectives = np.argmax(objectives_info)
        return all_centroids[argmax_objectives]


def plot_kmeans_given_centroids(centroids):
    X = load_data('train')
    distance_matrix = ((X - centroids[:, np.newaxis]) ** 2).sum(axis=2)
    assignments = np.argmin(distance_matrix, axis=0)
    fig, ax = plt.subplots()
    scatter_plot(X, title='Clustering of the training data - Degenerate case', colors=assignments, ax=ax)
    ax.plot(centroids[:, 0], centroids[:, 1], 'rx', mew=5, ms=10, label='centroids')
    ax.legend()


def train_gmm(K=4, var=1, d=2, spherical=True, old=False):
    # if old, we use ugly double loops instead of einstein summation
    X = load_data('train')
    Xtest = load_data('test')
    model = EMGMM(K=K, var=var, d=d, spherical=spherical)
    tau = model.fit(X, old=old)
    train_log_likelihood = model.log_likelihood(X)
    test_log_likelihood = model.log_likelihood(Xtest)
    return X, model.mu, model.sigma, tau.argmax(axis=1), train_log_likelihood, test_log_likelihood


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


def plot_gmm(K=4, var=1, d=2, spherical=True, old=False):
    # if old, we use ugly double loops instead of einstein summation
    X, means, covariances, assignment_array, train_L, test_L = train_gmm(K, var, d, spherical, old)
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter_plot(X, title='EM for the GMM with {}spherical covariance matrices'.format('non-' if not spherical else ''),
                 colors=assignment_array, ax=ax)
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

    return train_L, test_L









