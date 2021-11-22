import numpy as np
import pandas as pd
import itertools
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from typing import List

def clean_dataframe(df):
    data = df.copy(deep=True)
    solved, unsolved = split_solved(data)
    data = remove_col(solved, ['Situation', 'Incident', 'Ori', 'StateName'])
    data = fill_unknown(data)
    data = delete_val(data, ['OffSex', 'OffRace', 'VicSex', 'VicRace'], ['Unknown', 'Unknown', 'Unknown', 'Unknown'])
    data = clean_unk(data)
    data = del_agentype(data, 'Agentype')
    data = split_filedate(data)
    data = split_county_area(data)
    return data


# Split the dataframe into a solved and unsolved dataset
def split_solved(df):
    grouped = df.groupby(df.Solved)
    return grouped.get_group("Yes"), grouped.get_group("No")


# Fill in missing data (NaN) with Unknown value
def fill_unknown(df):
    return df.fillna("Unknown")


# Remove column(s)
def remove_col(df, cols):
    return df.drop(cols, axis=1)


# Delete rows with specific column value
def delete_val(df, cols, values):
    for i in range(len(cols)):
        df = df.drop(df[df[cols[i]] == values[i]].index)
    return df


# Everything that is unknown/undetermined/not specified/not reported/not determined, change to Unknown
def clean_unk(df):
    df.loc[df['OffAge'] == 999, 'OffAge'] = 'Unknown'
    df.loc[df['VicAge'] == 999, 'VicAge'] = 'Unknown'
    for col in df.columns:
        try:
            df.loc[df[col].str.contains("unknown|undetermined|not specified|not reported|not determined",
                                        regex=True), col] = 'Unknown'
        except:
            continue
    return df


# Delete Agency Type which is equal to 4 (unclear)
def del_agentype(df, col):
    return df.drop(df[df[col] == '4'].index)


# Split the FileDate column into three columns (TO BE FINISHED)
def split_filedate(df):
    df['FileDate'] = df['FileDate'].astype(str)
    df.insert(1, 'FileYear', df['FileDate'].map(lambda x : x[len(x)-4:len(x)-2]), True)
    df.insert(1, 'FileDay', df['FileDate'].map(lambda x : x[len(x)-6:len(x)-4]), True)
    df.insert(1, 'FileMonth', df['FileDate'].map(lambda x : x[0:len(x)-6]), True)
    df['FileYear'] = df['FileYear'].astype(str)
    df['FileMonth'] = df['FileMonth'].astype(str)
    df['FileDay'] = df['FileDay'].astype(str)
    df = df.drop(['FileDate'], axis=1)
    return df


# Converts all string values to numeric values (integer refering to a specific value)
def to_numeric(df, use_ordinal_encoder: bool = False, non_numeric_features: List[str] = []):
    df_numeric = df.copy()
    if not use_ordinal_encoder:

        for col in df.columns:
        # if(col in ['File Month','File Year','File Day']):
        #   df_numeric[col] = pd.to_numeric(df[col])
        #   print(col)
            if df[col].dtype == 'object':
                labels = df_numeric[col].unique().tolist()
                mapping = dict(zip(labels,range(len(labels))))
                df_numeric.replace({col: mapping},inplace=True)

        return df_numeric
    else:
        encoder = OrdinalEncoder()
        df_numeric[non_numeric_features] = pd.DataFrame(encoder.fit_transform(df_numeric[non_numeric_features]), columns=non_numeric_features)
        return df, encoder


# Split CNTYFIPS and MSA columns into County and Area
def split_county_area(df):
    df.insert(0, 'County', df['CNTYFIPS'].map(lambda x : str(x).split(',')[0]), True)
    df.insert(0, 'Area',df['MSA'].map(lambda x :  str(x).split(',')[0] ) ,True)
    df = df.drop(['CNTYFIPS'], axis=1)
    df = df.drop(['MSA'], axis=1)
    return df


# Split the FileDate column into three columns
def split_filedate(df):
    df['FileDate'] = df['FileDate'].astype(str)
    df.insert(0, 'FileYear', df['FileDate'].map(lambda x : x[len(x)-4:len(x)-2]), True)
    df.insert(0, 'FileDay', df['FileDate'].map(lambda x : x[len(x)-6:len(x)-4]), True)
    df.insert(0, 'FileMonth', df['FileDate'].map(lambda x : x[0:len(x)-6]), True)
    df['FileYear'] = df['FileYear'].astype(str)
    df['FileMonth'] = df['FileMonth'].astype(str)
    df['FileDay'] = df['FileDay'].astype(str)
    df = df.drop(['FileDate'], axis=1)
    return df


# Devide the dataset into two different dataframes representing the input and output features
def split_input_from_output(data, input_columns, output_columns):
    indices = []
    for column in input_columns:
        indices.append(data.columns.get_loc(column))
    input_data = data.iloc[:,np.asarray(indices)]
    indices = []
    for column in output_columns:
        indices.append(data.columns.get_loc(column))
    output_data = data.iloc[:,np.asarray(indices)]
    return input_data, output_data


# Example of how to get the datasets with in- and output features (including the current interpretation)
def get_current_input_and_output(data):
    input_columns = ['CNTYFIPS', 'State', 'Agency', 'Agentype', 'Homicide', 'VicAge', 'VicSex', 'VicRace', 'VicEthnic', 'Weapon', 'Subcircum', 'VicCount', 'MSA', 'Circumstance']
    output_columns = ['OffAge', 'OffSex', 'OffRace', 'OffEthnic', 'Relationship', 'OffCount']
    return split_input_from_output(data, input_columns, output_columns)


# Split dataframe into small train and test set, stratified on cols
def split_stratify(df, cols, train_frac, test_frac):
    # Get unique combinations of columns
    unique_vals = [np.unique(df[[col]].values) for col in cols]
    combinations = list(itertools.product(*unique_vals))

    # Create df for each combination and sample non-random from df
    train, test = [], []
    for combi in combinations:
        binned_df = df.loc[(df[cols[0]] == combi[0]) & (df[cols[1]] == combi[1]) & (df[cols[2]] == combi[2])]
        train.append(binned_df.sample(frac=train_frac, replace=True, random_state=1))
        test.append(binned_df.sample(frac=test_frac, replace=True, random_state=1))

    # Return training df and test df
    return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True)

# generate column that has binned age values
def bin_age(df, age_col_name):
    bins = [0,2,14,18,22,30,40,50,60,70,80,100]
    labels = ['0-2','3-14', '15-18', '19-22', '23-30', '30s','40s','50s','60s','70s','80+']
    try:
        # cannot cut the df if age_col contains non-numeric dtypes
        binned_series = pd.cut(df[age_col_name], bins = bins, labels = labels)
    except TypeError:
        # replace 'Unkown' entries with age of 999
        bins = [0,2,14,18,22,30,40,50,60,70,80,100,1000]
        labels = ['0-2','3-14', '15-18', '19-22', '23-30', '30s','40s','50s','60s','70s','80+', 'Unknown']
        df[age_col_name].replace(to_replace='Unknown', value = 999, inplace = True)
        binned_series = pd.cut(df[age_col_name], bins = bins, labels = labels)
    return binned_series.astype(str)