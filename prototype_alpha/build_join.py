"""
This file will build the initial csv files.

use df.iloc[:,:-1] to use all but the last column

For the train_users.csv & test_users.csv:
----------------------------------------
  - id: will always be there
  - all others are categorical features
**- country_destination: what we're predicting(!)
"""
import pandas as pd
import numpy as np
from sklearn.externals import joblib

timestamps = ['date_account_created','timestamp_first_active','date_first_booking']

# =======================  For Prototypes Alpha
train_users = pd.read_csv("train_users.csv")
test_users = pd.read_csv("test_users.csv")
final_train = []; final_test = [];

# Issue with 'first_browser' feature
a = list(np.unique(train_users['first_browser']));
b = list(np.unique(test_users['first_browser']));
extras = list(set(a) - set(b))

# Going to use pd.get_dummies() for this all
print("Train Users")
for column in train_users:
    print("Column: " + column)
    if column == 'id' or column == "country_destination" or column == 'age' or column == 'signup_flow':
        final_train.append(train_users[column]);
    elif column in timestamps:
        final_train.append(pd.to_datetime(train_users[column]).astype("int64"));
    elif column == 'first_browser':
        final_train.append(pd.get_dummies(train_users[column],dummy_na=True).drop(extras,axis=1));
    else:
        final_train.append(pd.get_dummies(train_users[column],dummy_na=True));

print("Onto Testing Users")
for column in test_users:
    print("Column: " + column);
    if column == 'id' or column == 'age' or column == 'signup_flow':
        final_test.append(test_users[column]);
    elif column == "signup_method":
        final_test.append(pd.get_dummies(test_users[column],dummy_na=True).drop(['google'],axis=1));
    elif column in timestamps:
        final_test.append(pd.to_datetime(test_users[column]).astype("int64"));
    else:
        final_test.append(pd.get_dummies(test_users[column],dummy_na=True));

training = pd.concat(final_train,axis=1)
testing = pd.concat(final_test,axis=1)

trextras = list(set(training.columns) - set(testing.columns))  # Mucking up b/c of pd.concat most likely
trextras = [x for x in trextras if x != "country_destination"]
tsextras = list(set(testing.columns) - set(training.columns))  # Mucking up b/c of pd.concat most likely
training.drop(trextras,axis=1,inplace=True)
testing.drop(tsextras,axis=1,inplace=True)

# Impute values on the 'age' column
training['age'] = training['age'].fillna( training['age'].median() )
testing['age'] = testing['age'].fillna( testing['age'].median() )

print(training.shape)
print(testing.shape)

# Output 
training.to_csv("protoAlpha_training.csv",index=False);
testing.to_csv("protoAlpha_testing.csv",index=False);


