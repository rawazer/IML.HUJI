from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
from datetime import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.drop(columns=['id', 'lat', 'long']) # will work with zipcode
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S', errors='coerce')
    df['date'] = df['date'].apply(dt.toordinal)

    df = df.dropna()
    for col in ['date', 'price', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']:
        df = df.drop(df[df[col]<10].index)

    df = pd.get_dummies(df, columns=['zipcode']) # we'll encode them with one-hot

    X = df.drop(columns=['price'])
    y = df['price']
    return X, y
    


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col in X:
        corr = X[col].cov(y) / (X[col].std() * y.std())
        fig = px.scatter(x=[col], y=[corr], range_y=[0,1], 
                        title=col + ", Pearson correlation = " + str(corr),
                        labels={"x": "feature", "y": "correlation"})
        fig.write_image(output_path + "/" + col + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "./feature_eval_plots")

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    percentages = np.arange(0.1, 1, 0.01)
    mean_loss = np.empty(percentages.shape)
    std_loss = np.empty(percentages.shape)
    lin_reg = LinearRegression(include_intercept=True)
    frames = []
    for (ind, p) in enumerate(percentages):
        losses = []
        for i in range(10):
            cur_X_train = X_train.sample(frac = p, random_state=i).to_numpy()
            cur_y_train = y_train.sample(frac = p, random_state=i).to_numpy()
            lin_reg.fit(cur_X_train, cur_y_train)
            loss = lin_reg.loss(X_test.to_numpy(), y_test.to_numpy())
            losses.append(loss)
        mean_loss[ind] = np.mean(losses)
        std_loss[ind] = np.std(losses)
    fig = go.Figure(
        data=[
            go.Scatter(x=percentages, y=mean_loss, name="loss", mode="markers+lines", line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
            go.Scatter(x=percentages, y=mean_loss + 2 * std_loss, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
            go.Scatter(x=percentages, y=mean_loss - 2 * std_loss, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False)]
    )
    fig.update_xaxes(title_text="Percentage")
    fig.update_yaxes(title_text="Mean loss")
    fig.update_layout(title="Mean loss as a function of percentage of training data used, with std")
    fig.show()
