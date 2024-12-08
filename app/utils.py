import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def load_data(file_path):
    try:
        df_3 = pd.read_csv(file_path)
        return df_3
    except pd.errors.ParserError:
        return None


def perform_seasonal_decomposition(df_3, period):
    decomposition = seasonal_decompose(df_3["GHI"], model='additive', period=period)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    return trend, seasonal, residual


def perform_box_plot_analysis(df_3, variables):
    boxplot_data = df_3[variables]
    fig, ax = plt.subplots()
    sns.boxplot(data=boxplot_data, ax=ax)
    ax.set_ylabel("Value")
    return fig


def perform_correlation_analysis(df_3, variable1, variable2):
    correlation = df_3[variable1].corr(df_3[variable2])
    scatter_plot = plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_3[variable1], y=df_3[variable2])
    plt.xlabel(variable1)
    plt.ylabel(variable2)
    plt.title("Scatter Plot")
    plt.grid(True)
    return correlation, scatter_plot