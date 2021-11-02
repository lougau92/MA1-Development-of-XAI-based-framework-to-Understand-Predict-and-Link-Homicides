# removing obvious redundant or useless features
def feature_cleaning(df):
    df = df.copy()
    # Feature Cleaning
    df.insert(1, 'County', df['CNTYFIPS'].map(lambda x : str(x).split(',')[0]), True)
    df.drop("CNTYFIPS",axis = 1, inplace= True) # redundant, onlyy keeping the county part
    df.drop("StateName",axis = 1, inplace= True)  # redundant, contains state
    df.drop("Ori",axis = 1, inplace= True)  # redundant, contains state + county + agentcy nb
    df.insert(3, 'Area',df['MSA'].map(lambda x :  str(x).split(',')[0] ) ,True)
    df.drop('MSA',axis = 1,inplace=True) # rendundant only the state, keeping the metropolitant statistical area

    # split File Date in Day-Month-Year columns
    nan_indexes = df["FileDate"].isna()
    df['FileDate'].fillna(0,inplace = True)
    df['FileDate'] = df['FileDate'].astype(int)
    df['FileDate'] = df['FileDate'].astype(str)
    df.insert(1, 'File Year', df['FileDate'].map(lambda x : x[len(x)-2:len(x)]), True)
    df.insert(1, 'File Day', df['FileDate'].map(lambda x : x[len(x)-4:len(x)-2]), True)
    df.insert(1, 'File Month', df['FileDate'].map(lambda x : x[0:len(x)-4]), True)
    df.loc[nan_indexes,"File Year"] = np.NaN
    df.loc[nan_indexes,"File Day"] = np.NaN
    df.loc[nan_indexes,"File Month"] = np.NaN
    df.drop("FileDate",axis = 1,inplace = True)

    df.drop("Incident",axis = 1,inplace = True) # incient number of the month, for a specific county useless for our application

    df.drop("Situation",axis = 1,inplace = True) # redundant, info already contained in OffCount and VicCount

    # drop source ?
    return df

# handling some data values inconsistencies
def values_cleaning(df):
    df = df.copy()

    df['OffAge'].replace(999, np.NaN, inplace=True) # replacing 999 ages with Nan
    df = df[df["Agentype"]!='4'] # removing 4 4s typo?

    return df

# removing all data instances containing at least one missing value
def missing_values_removing(df, drop_Subcircum = False):
    df = df.copy()
    
    if(drop_Subcircum): df = df.drop(["Subcircum"],axis = 1) # more than 80% of values are missing for Subcircum

    df = df.dropna()
    
    unknow_synonyms = "nknown|undetermined|not specified" # please not that this will be used as a regex expression, hence "nknown" is convering "Unknown" and "unknown", "|" means or

    for col in df.columns:
        try:
            df = df[df[col].str.contains(unknow_synonyms,regex = True)==False]
        except:
                a=0
                # print("exception column ",col)
    return df

# converts all the string values to numeric values (integer refering to a specific value)
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