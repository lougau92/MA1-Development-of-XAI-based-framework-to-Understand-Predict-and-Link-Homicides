import numpy as np
from numpy.core.numeric import indices
import pandas as pd
import plotly.figure_factory as ff
from sklearn.metrics import normalized_mutual_info_score
import seaborn as sns
from scipy import stats
from typing import List, Optional

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

    p_rounded = np.around(p_values, decimals=4)
    fig = ff.create_annotated_heatmap(p_values, 
                                        x=data.columns.tolist(), 
                                        y=data.columns.tolist(),
                                        annotation_text=p_rounded)
    fig.update_xaxes(side="top")
    fig.update_layout( title_text='p_values of Chi-Squared test for independence',
                        autosize=False,
                        width=2000,
                        height=2000
                        )
    fig.show()

    if return_df:
        return pd.DataFrame(p_values, index=data.columns, columns=data.columns)

def heatmap_mutual_info(data: pd.DataFrame, return_df: bool = False, figsize: tuple = (30, 30)) -> Optional[pd.DataFrame]:
    num_features = len(data.columns)
    mutual_info_scores = np.zeros(shape=(num_features, num_features))
    for i in range(num_features):
        for j in range(i, len(data.columns)):
            mutual_info = normalized_mutual_info_score(data.iloc[:, i], data.iloc[:, j])
            mutual_info_scores[i][j] = mutual_info
            mutual_info_scores[j][i] = mutual_info

    mi_rounded = np.around(mutual_info_scores, decimals=4)
    fig = ff.create_annotated_heatmap(mutual_info_scores,
                    x=data.columns.tolist(),
                    y=data.columns.tolist(),
                    annotation_text=mi_rounded
                )
    #fig.update_xaxes(side="top")
    fig.update_layout( title_text='Mutual Information scores',
                        autosize=False,
                        width=2000,
                        height=2000
                        )
    fig.show()

    #plt.figure(figsize=figsize)
    #heat_map = sns.heatmap(df, linewidth = 1 , robust = True, annot = True)
    #plt.title("Mutual Information scores")
    #plt.show()
    if return_df:
        return pd.DataFrame(mutual_info_scores, index=data.columns, columns=data.columns)