import numpy as np
import pandas as pd


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
def split_col(df):
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
def to_numeric(df):
    df = df.copy()
    df_numeric = df.copy()

    for col in df.columns:
      # if(col in ['File Month','File Year','File Day']):
      #   df_numeric[col] = pd.to_numeric(df[col])
      #   print(col)
        if df[col].dtype == 'object':
            labels = df_numeric[col].unique().tolist()
            mapping = dict( zip(labels,range(len(labels))) )
            df_numeric.replace({col: mapping},inplace=True)

    return df_numeric

# Split CNTYFIPS and MSA columns into County and Area
def split_county_area(df):
    df.insert(0, 'County', df['CNTYFIPS'].map(lambda x : str(x).split(',')[0]), True)
    df.insert(0, 'Area',df['MSA'].map(lambda x :  str(x).split(',')[0] ) ,True)
    df = df.drop(['CNTYFIPS'], axis=1)
    df = df.drop(['MSA'], axis=1)
    return df