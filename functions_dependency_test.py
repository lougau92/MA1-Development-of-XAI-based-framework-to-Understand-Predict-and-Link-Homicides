import numpy as np
from numpy.core.numeric import indices
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List
plt.style.use("seaborn")

def p_of_chi_squared(feature_1: pd.Series, feature_2: pd.Series) -> float:
    """Performs the chi-squared test of independence between the two passed features, returns the p-value

    Args:
        feature_1 (pd.Series): All observations for feature no. 1 to be used in the test
        feature_2 (pd.Series): All observations for feature no. 2 to be used in the test

    Returns:
        float: p-value of the test
    """
    return stats.chi2_contingency(pd.crosstab(feature_1, feature_2))[1]

def find_dependent_chi(data: pd.DataFrame, significance_level: float) -> pd.DataFrame:
    """Performs the chi-squared test of independen between all columns of the data frame, returns siginificant pairs

    Args:
        data (pd.DataFrame): DataFrame that only contains feature variables
        significance_level (float): bounded between [0,1]

    Returns:
        pd.DataFrame: DataFrame with 3 columns: feature_1 and feature_2 are the name of the variable pair, p_value the result of the chi-squared test for the feature pair.
        Only lists pairs for which the Chi-squared test gave a p-value <= significance_level
    """
    assert 0 <= significance_level <= 1, 'Invalid significance level, must be in range [0,1]'

    feature_1 = []
    feature_2 = []
    p_value = []
    num_features = len(data.columns)
    for i in range(num_features):
        feature_1.extend([data.columns[i]]*(num_features-(i+1)))
        for j in range(i+1, len(data.columns)):
            feature_2.append(data.columns[j])
            p_value.append(p_of_chi_squared(data.iloc[:, i], data.iloc[:, j]))
    df_dict = {'Feature 1': feature_1, 'Feature 2': feature_2, 'p-value': p_value}
    df = pd.DataFrame(df_dict)
    insignificants = df[df['p-value'] > significance_level].index
    df.drop(insignificants, inplace=True)
    return df.sort_values(by='p-value')

def p_values_chi(data: pd.DataFrame) -> pd.DataFrame:
    num_features = len(data.columns)
    p_values = np.zeros(shape=(num_features, num_features))
    for i in range(num_features):
        for j in range(i, len(data.columns)):
            p = p_of_chi_squared(data.iloc[:, i], data.iloc[:, j])
            p_values[i][j] = p
            p_values[j][i] = p
    return pd.DataFrame(p_values, index=data.columns, columns=data.columns)

def heatmap_chi(data: pd.DataFrame, return_df: bool = False, figsize: tuple = (30, 30)) -> Optional[pd.DataFrame]:
    num_features = len(data.columns)
    p_values = np.zeros(shape=(num_features, num_features))
    for i in range(num_features):
        for j in range(i, len(data.columns)):
            p = p_of_chi_squared(data.iloc[:, i], data.iloc[:, j])
            p_values[i][j] = p
            p_values[j][i] = p

    df = pd.DataFrame(p_values, index=data.columns, columns=data.columns)
    plt.figure(figsize=figsize)
    heat_map = sns.heatmap(df, linewidth = 1 , robust = True, annot = True)
    plt.title("p_values of Chi-Squared test for independence")
    plt.show()
    if return_df:
        return df