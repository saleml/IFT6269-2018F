import numpy as np
from matplotlib import pyplot as plt
import os
from fisher_lda import FDA
from logistic_regression import LogReg
from linear_regression import LinReg
from qda import QDA

ROOT = 'hwk2data'


def sigmoid(z):
    return 1. / (1 + np.exp(- z))


def file_to_dict(filename, bias=False):
    with open(os.path.join(ROOT, filename)) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [x.split() for x in content]
    X = np.array([[float(element) for element in x[:-1] + ([1] if bias else [])] for x in content])
    Y = np.array([float(x[-1]) for x in content])
    return {'X': X, 'Y': Y}


def scatter_plot(XY_dict, title='', ax=None, **kwargs):
    X = XY_dict['X']
    Y = XY_dict['Y']
    if ax is None:
        ax = plt
    ax.scatter(X[:, 0], X[:, 1], c=Y, **kwargs)
    ax.set_title(title)


def plot_datasets():
    rows, cols = 3, 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    k = 0
    for filename in sorted(os.listdir(ROOT)):
        i, j = k // cols, k % cols
        XY_dict = file_to_dict(filename)
        scatter_plot(XY_dict, title=filename, ax=axes[i, j])
        k += 1


def models(modeltype='FDA'):
    '''
    Returns a dictionary of (trained) models. One per dataset.
    '''

    models_dict = dict()

    for filename in sorted(os.listdir(ROOT)):
        if filename.endswith('train'):
            dataset = filename.split('.')[0]
            XY_dict_train = file_to_dict(filename)
            X = XY_dict_train['X']
            Y = XY_dict_train['Y']

            if modeltype == 'FDA':
                model = FDA()
            elif modeltype == 'LogReg':
                model = LogReg()
            elif modeltype == 'LinReg':
                model = LinReg()
            elif modeltype == 'QDA':
                model = QDA()
            else:
                raise ValueError("model not implemented")

            models_dict[dataset] = model

            model.fit(X, Y)

    return models_dict


def show_results(models, quadratic=False,):
    '''
    Plot decision boundaries on train and test. Show the coefficients of the separator, and the classification error.
    '''

    rows, cols = 3, 2
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    k = 0

    for filename in sorted(os.listdir(ROOT)):
        dataset = filename.split('.')[0]
        model = models[dataset]
        XY_dict = file_to_dict(filename)
        X = XY_dict['X']
        Y = XY_dict['Y']

        decision_boundary, coefficients = model.decision_boundary(X)
        i, j = k // cols, k % cols
        ax = axes[i, j] if 1 not in (rows, cols) else axes[i + j]
        scatter_plot(XY_dict, title=filename, ax=ax)
        if not quadratic:
            ax.plot(*decision_boundary, label='Decision boundary (obtained from train set)')
        else:
            x0, x1 = np.meshgrid(*decision_boundary)
            a, b, c, d, e, f = coefficients
            ax.contour(x0, x1, (a*x0**2 + b*x0*x1 + c*x1**2 + d*x0 + e*x1 + f), [0])
        x0_MIN, x0_MAX = np.min(X[:, 0]), np.max(X[:, 0])
        x1_MIN, x1_MAX = np.min(X[:, 1]), np.max(X[:, 1])
        ax.set_xlim([x0_MIN - 5, x0_MAX + 5])
        ax.set_ylim([x1_MIN - 5, x1_MAX + 5])

        if not quadratic:
            ax.legend()
            ax.set_title(ax.title.get_text() + ', error: {:.2f}%'.format(100 * model.classification_error(X, Y)))
            a, b, c = coefficients
            ax.set_title(ax.title.get_text() + '\nseparator equation: ${:.2f}x_0 + {:.2f}x_1 + {:.2f} = 0$ or $x_1 = {:.2f}x_0 + {:.2f}$'.format(*coefficients, -a / b, -c / b))
        else:
            ax.set_title(ax.title.get_text() + ', error: {:.2f}% - Conics are the decision boundaries'.format(100 * model.classification_error(X, Y)))
            ax.set_title(ax.title.get_text() + '\nseparator equation: ${:.2f}x_0^2 + {:.2f}x_0 x_1 + {:.2f}x_1^2 + {:.2f}x_0 + {:.2f}x_1 + {:.2f}=0$'.format(*coefficients))
        k += 1
