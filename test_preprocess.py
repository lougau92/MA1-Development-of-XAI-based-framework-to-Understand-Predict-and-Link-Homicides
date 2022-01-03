import pandas as pd

from preprocess import clean_dataframe,split_stratify
# https://scikit-learn.org/stable/modules/svm.html
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('Murder_Data.zip', index_col=0, compression='zip',low_memory=False)
cleaned_data = clean_dataframe(raw_data)
#get a stratisfied subset to test the algorithms on
#train,test = split_stratify(...)
#get subset of 20% of total data
cleaned_data, _ = split_stratify(cleaned_data,['OffSex', 'OffRace', 'OffEthnic'],0.20,0.01)
#get 75% for training (of subset)
cleaned_train,cleaned_rest = split_stratify(cleaned_data,['OffSex', 'OffRace', 'OffEthnic'],0.70,0.30)
#get 83% for testing (results in 25% test and 5% val)
cleaned_test,cleaned_val = split_stratify(cleaned_rest,['OffSex', 'OffRace', 'OffEthnic'],0.83333,0.16667)

print(cleaned_data.shape)
print(cleaned_train.shape)
print(cleaned_test.shape)
print(cleaned_val.shape)

print(cleaned_train)
print(cleaned_rest)
print(cleaned_val)