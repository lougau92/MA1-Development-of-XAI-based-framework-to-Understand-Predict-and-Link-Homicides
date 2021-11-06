import numpy as np
import pandas as pd


def clean_dataframe(df):
    data = df.copy(deep=True)
    solved, unsolved = split_solved(data)
    data = fill_unknown(solved)
    data = remove_col(data, ['StateName', 'Situation', 'Incident'])
    data = delete_val(data, ['OffSex', 'OffRace', 'VicSex', 'VicRace'], ['Unknown', 'Unknown', 'Unknown', 'Unknown'])
    data = clean_unk(data)
    data = del_agentype(data, 'Agentype')
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
def split_col(df):
    #     df[['FileMonth', 'FileDay', 'FileYear']] = df['AB'].str.split(' ', 1, expand=True)
    #     df['FileDate'] = df['FileDate'].astype(int)
    #     df['FileDate'] = df['FileDate'].astype(str)
    df.insert(1, 'File Year', df['FileDate' != 'Unknown'].map(lambda x: x[len(x) - 2:len(x)]), True)
    df.insert(1, 'File Day', df['FileDate'].map(lambda x: x[len(x) - 4:len(x) - 2]), True)
    df.insert(1, 'File Month', df['FileDate'].map(lambda x: x[0:len(x) - 4]), True)
    #df.loc[nan_indexes, "File Year"] = np.NaN
    #df.loc[nan_indexes, "File Day"] = np.NaN
    #df.loc[nan_indexes, "File Month"] = np.NaN
