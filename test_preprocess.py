import pandas as pd

from preprocess import clean_dataframe,get_train_test_val
# https://scikit-learn.org/stable/modules/svm.html
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('Murder_Data.zip', index_col=0, compression='zip',low_memory=False)
cleaned_data = clean_dataframe(raw_data)
#get a stratisfied subset to test the algorithms on
#train,test = split_stratify(...)
#get subset of 20% of total data

cleaned_train,cleaned_test,cleaned_val = get_train_test_val(cleaned_data,['OffSex', 'OffRace', 'OffEthnic'])

print(cleaned_data.shape)
print(cleaned_train.shape)
print(cleaned_test.shape)
print(cleaned_val.shape)

print(cleaned_train)
print(cleaned_test)
print(cleaned_val)