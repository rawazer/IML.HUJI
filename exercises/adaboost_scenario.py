import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada = AdaBoost(DecisionStump, n_learners)
    ada.fit(train_X, train_y)
    test_err, train_err = [], []
    for t in range(1, n_learners, 1):
        train_err.append(ada.partial_loss(train_X, train_y, t))
        test_err.append(ada.partial_loss(test_X, test_y, t))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(1, n_learners, n_learners), y=train_err, 
                        mode='lines', name='train'))
    fig.add_trace(go.Scatter(x=np.linspace(1, n_learners, n_learners), y=test_err, 
                        mode='lines', name='test'))
    fig.update_layout(title="Loss of AdaBoost against train and test for number of models",
            xaxis_title="Number of models used",
            yaxis_title="Loss")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    symbols_test = test_y.astype(str)
    symbols_test[symbols_test == '1.0'] = 'cross-thin'
    symbols_test[symbols_test == '-1.0'] = 'line-ew'
    for t in T:
        def cur_pred(X):
            return ada.partial_predict(X, t)
        fig = go.Figure()
        fig.add_trace(decision_surface(cur_pred, lims[0, :], lims[1, :]))
        fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
            mode='markers', marker=dict(
            symbol=symbols_test,
            size=np.abs(test_y) * 3
        )))
        fig.update_layout(title="Decision surfaces of AdaBoost using " + str(t) + " models")
        fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_t = np.argmin(test_err) + 1
    def cur_pred(X):
        return ada.partial_predict(X, best_t)
    fig = go.Figure()
    fig.add_trace(decision_surface(cur_pred, lims[0, :], lims[1, :]))
    fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
            mode='markers', marker=dict(
                symbol=symbols_test,
                size=np.abs(test_y) * 3
            )))
    fig.update_layout(title="Decision surfaces of AdaBoost using best amount of models = " + str(best_t) + ",\nGot an error of " + str(test_err[best_t - 1]))
    fig.show()

    # Question 4: Decision surface with weighted samples
    def cur_pred(X):
        return ada.partial_predict(X, n_learners)
    D = (ada.D_ / np.max(ada.D_)) * 10
    symbols_train = train_y.astype(str)
    symbols_train[symbols_train == '1.0'] = 'cross-thin'
    symbols_train[symbols_train == '-1.0'] = 'line-ew'
    fig = go.Figure()
    fig.add_trace(decision_surface(cur_pred, lims[0, :], lims[1, :]))
    fig.add_trace(go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                mode='markers', marker=dict(
                    symbol=symbols_train,
                    size=D
                )))
    fig.update_layout(title="Decisions of last ensemble")
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0, 250)
    fit_and_evaluate_adaboost(0.4, 250)

