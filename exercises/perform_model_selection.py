from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
import sklearn
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.random.uniform(-1.2, 2.0, n_samples)
    x_noise = np.random.normal(0, noise, n_samples)
    y = (x+3)*(x+2)*(x+1)*(x-1)*(x-2)+x_noise

    train_portion = 2/3
    x = pd.DataFrame(data=x)
    y = pd.Series(data=y)
    X_train, y_train, X_test, y_test = split_train_test(x, y, train_portion)
    X_train = X_train.to_numpy().flatten()
    X_test = X_test.to_numpy().flatten()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train.T, y=y_train.T, mode="markers", name="Train", marker=dict(color="Blue")))
    fig.add_trace(go.Scatter(x=X_test.T, y=y_test.T, mode="markers", name="Test", marker=dict(color="Red")))
    fig.update_layout(title="Train/Test split of generated data")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    polynomials_amt = 11
    fold_amt = 5

    average_train_errors = np.empty(polynomials_amt)
    average_validation_errors = np.empty(polynomials_amt)

    for k in range(polynomials_amt):
        train_errors = np.empty(fold_amt-1)
        val_errors = np.empty(fold_amt-1)
        split_x = np.array_split(X_train, fold_amt)
        split_y = np.array_split(y_train, fold_amt)
        
        polyfit = PolynomialFitting(k)

        for i in range(fold_amt-1):
            cur_val_x = split_x[i]
            cur_train_x = np.concatenate((split_x[:i] + split_x[i+1:]))
            cur_val_y = split_y[i]
            cur_train_y = np.concatenate((split_y[:i] + split_y[i+1:]))

            polyfit.fit(cur_train_x, cur_train_y)
            train_errors[i] = polyfit.loss(cur_train_x, cur_train_y)
            val_errors[i] = polyfit.loss(cur_val_x, cur_val_y)
        
        average_train_errors[k] = np.average(train_errors)
        average_validation_errors[k] = np.average(val_errors)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(polynomials_amt), y=average_train_errors,
                mode="markers+lines", name="Avg. train error"))
    fig.add_trace(go.Scatter(x=np.arange(polynomials_amt), y=average_validation_errors,
                mode="markers+lines", name="Avg. validation error"))
    fig.update_layout(title="Average train and validation errors for degrees of polynome fit",
            xaxis_title="Polynomial degree", yaxis_title="MSE loss")
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error

    best_k = np.argmin(average_validation_errors)
    best_polyfit = PolynomialFitting(best_k)
    best_polyfit.fit(X_train, y_train)
    print("Best polynomial degree found: ", best_k)
    print("Test loss of matching model: ", best_polyfit.loss(X_test, y_test))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    diabetes_X = diabetes_X.sample(n=n_samples, random_state=2022)
    diabetes_y = diabetes_y.sample(n=n_samples, random_state=2022)
    lam_range = np.linspace(1/n_evaluations, 5 + 1/n_evaluations, n_evaluations)

    train_portion = 2/3
    X_train, y_train, X_test, y_test = split_train_test(diabetes_X, diabetes_y, train_portion)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    average_train_errors_lasso = np.empty(n_evaluations)
    average_validation_errors_lasso = np.empty(n_evaluations)
    average_train_errors_ridge = np.empty(n_evaluations)
    average_validation_errors_ridge = np.empty(n_evaluations)
    fold_amt = 5

    for i in range(n_evaluations):
        lam = lam_range[i]
        train_errors_lasso = np.empty(fold_amt-1)
        val_errors_lasso = np.empty(fold_amt-1)
        train_errors_ridge = np.empty(fold_amt-1)
        val_errors_ridge = np.empty(fold_amt-1)
        split_x = np.array_split(X_train, fold_amt)
        split_y = np.array_split(y_train, fold_amt)

        lasso = Lasso(alpha=lam)
        ridge = RidgeRegression(lam=lam)

        for j in range(fold_amt-1):
            cur_val_x = split_x[j]
            cur_train_x = np.concatenate((split_x[:j] + split_x[j+1:]))
            cur_val_y = split_y[j]
            cur_train_y = np.concatenate((split_y[:j] + split_y[j+1:]))

            lasso.fit(cur_train_x, cur_train_y)
            ridge.fit(cur_train_x, cur_train_y)
            train_errors_lasso[j] = mean_square_error(lasso.predict(cur_train_x), cur_train_y)
            val_errors_lasso[j] = mean_square_error(lasso.predict(cur_val_x), cur_val_y)
            train_errors_ridge[j] = ridge.loss(cur_train_x, cur_train_y)
            val_errors_ridge[j] = ridge.loss(cur_val_x, cur_val_y)
        
        average_train_errors_lasso[i] = np.average(train_errors_lasso)
        average_validation_errors_lasso[i] = np.average(val_errors_lasso)
        average_train_errors_ridge[i] = np.average(train_errors_ridge)
        average_validation_errors_ridge[i] = np.average(val_errors_ridge)
    
    fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["Lasso", "Ridge"])
    fig.add_trace(go.Scatter(x=lam_range, y=average_train_errors_lasso,
                mode="lines", name="Avg. train error of Lasso"), row=1, col=1)
    fig.add_trace(go.Scatter(x=lam_range, y=average_validation_errors_lasso,
                mode="lines", name="Avg. validation error of Lasso"), row=1, col=1)
    fig.add_trace(go.Scatter(x=lam_range, y=average_train_errors_ridge,
                mode="lines", name="Avg. train error of Ridge"), row=1, col=2)
    fig.add_trace(go.Scatter(x=lam_range, y=average_validation_errors_ridge,
                mode="lines", name="Avg. validation error of Ridge"), row=1, col=2)
    fig.update_xaxes(title_text="Regularization parameter", row=1, col=1)
    fig.update_xaxes(title_text="Regularization parameter", row=1, col=2)
    fig.update_yaxes(title_text="MSE loss", row=1, col=1)
    fig.update_yaxes(title_text="MSE loss", row=1, col=2)
    fig.update_layout(title_text="Average train and validation errors for different regularization parameters")
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model

    best_lasso_lambda = lam_range[np.argmin(average_validation_errors_lasso)]
    best_ridge_lambda = lam_range[np.argmin(average_validation_errors_ridge)]

    best_lasso = Lasso(alpha=best_lasso_lambda)
    best_ridge = RidgeRegression(lam=best_ridge_lambda)
    ls = LinearRegression()

    best_lasso.fit(X_train, y_train)
    best_ridge.fit(X_train, y_train)
    ls.fit(X_train, y_train)

    lasso_test_err = mean_square_error(best_lasso.predict(X_test), y_test)
    ridge_test_err = best_ridge.loss(X_test, y_test)
    ls_test_err = ls.loss(X_test, y_test)
    print("Lasso test error for best lambda, ", best_lasso_lambda, ", is ", lasso_test_err)
    print("Ridge test error for best lambda, ", best_ridge_lambda, ", is ", ridge_test_err)
    print("Least squares test error is ", ls_test_err)

if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
