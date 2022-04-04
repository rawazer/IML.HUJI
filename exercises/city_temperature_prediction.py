import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=[2])
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df = df.dropna()
    df = df[df["Temp"] > -10]
    X = df.drop(columns="Temp")
    y = df["Temp"]
    return X, y


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X, y = load_data("datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    X["Year"] = X["Year"].astype(str)
    X_isr = X[X["Country"]=="Israel"]
    y_isr = y[X["Country"]=="Israel"]
    fig = px.scatter(x=X_isr["DayOfYear"], y=y_isr, color=X_isr["Year"],
                     title="Temperature in israel per day, divided to years by color",
                     labels={"x":"Day of year", "y":"Temperature"})
    fig.show()
    X["Year"] = X["Year"].astype(np.int64)
    df_months = pd.concat([X_isr["Month"], y_isr], axis=1)
    df_months = df_months.groupby("Month").agg("std")
    fig = px.bar(df_months, x=df_months.index, y="Temp",
                 title="Standard deviation of the temperature in Israel by month",
                 labels={"Temp":"Standard deviation of temperature"})
    fig.show()

    # Question 3 - Exploring differences between countries
    df_countries = pd.concat([X["Country"], X["Month"], y], axis=1)
    df_countries = df_countries.groupby(["Country", "Month"]).agg(mean=("Temp", "mean"), std=("Temp", "std")).reset_index()
    fig = px.line(df_countries, x="Month", y="mean", error_y="std", color="Country",
                  title="Average temperature of countries by month, with standard deviation",
                  labels={"mean":"Average Temperature"})
    fig.show()

    # Question 4 - Fitting model for different values of `k`

    X_train, y_train, X_test, y_test = split_train_test(X_isr["DayOfYear"], y_isr)
    losses = []
    for k in range(1, 11, 1):
        poly_reg = PolynomialFitting(k)
        poly_reg.fit(X_train.to_numpy(), y_train.to_numpy())
        losses.append(np.round(poly_reg.loss(X_test.to_numpy(), y_test.to_numpy()), decimals=2))


    print(losses)
    fig = px.bar(x=[i for i in range(1, 11, 1)], y=losses,
                 title="Losses of polynomial fitting with different k",
                 labels={"x":"Degrees of polynomials", "y":"Loss"})
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    X_jord = X[X["Country"]=="Jordan"]
    y_jord = y[X["Country"]=="Jordan"]
    X_sa = X[X["Country"]=="South Africa"]
    y_sa = y[X["Country"]=="South Africa"]
    X_ntl = X[X["Country"]=="The Netherlands"]
    y_ntl = y[X["Country"]=="The Netherlands"]

    losses = []
    Xs = [X_jord["DayOfYear"], X_sa["DayOfYear"], X_ntl["DayOfYear"]]
    Ys = [y_jord, y_sa, y_ntl]

    poly_reg = PolynomialFitting(5)
    poly_reg.fit(X_isr["DayOfYear"].to_numpy(), y_isr.to_numpy())
    for i in range(3):
        losses.append(poly_reg.loss(Xs[i].to_numpy(), Ys[i].to_numpy()))
    
    Countries = ["Jordan", "South Africa", "The Netherlands"]

    fig = px.bar(x=Countries, y=losses,
                title="Losses of polynomial fitting to Israel for different countries",
                labels={"x":"County", "y":"Loss"})
    fig.show()