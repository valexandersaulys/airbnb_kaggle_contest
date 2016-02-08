"""
Take 1 on the RandomForest, predicting for country_destinations.
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

training = pd.read_csv("protoAlpha_training.csv")
testing = pd.read_csv("protoAlpha_testing.csv")

X = training.iloc[:,1:-1].values
y = training['country_destination'].values
x_train,x_valid,y_train,y_valid = train_test_split(X,y,test_size=0.3,random_state=None)

# LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_train);
y_train = le.transform(y_train);
y_valid = le.transform(y_valid);


# Train classifier
import xgboost as xgb
xg_train = xgb.DMatrix(x_train,label=y_train);
xg_valid = xgb.DMatrix(x_valid,label=y_valid);

# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'  # can be 'multi:softmax' or 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.9
param['max_depth'] = 100
param['gamma'] = 0.1
param['silent'] = 0   # 1 means silent mode
param['nthread'] = 5
param['num_class'] = len(np.unique(y_train).tolist());
param['booster'] = 'gbtree'  # default is 'gbtree'
param['subsample'] = 1.0  # default is 1.0
param['base_score'] = 0.5   # default is 0.5

# Train & Get validation data
num_round = 10
clf = xgb.train(param, xg_train, num_round);
#clf = xgb.cv(param, xg_train, num_round);

# get predictions
y_preds = clf.predict( xg_valid );


# Run Predictions
from sklearn.metrics import confusion_matrix, accuracy_score
print( confusion_matrix(y_valid,y_preds) );
print( "Accuracy: %f" % (accuracy_score(y_valid,y_preds)) );
f = open('xgboost_take7.txt', 'w')
f.write( str(confusion_matrix(y_valid,y_preds)) );
f.write( "\nAccuracy: %f" % (accuracy_score(y_valid,y_preds)) );
f.write( str(param) );

# Now on to final submission
xg_test = xgb.DMatrix(testing.iloc[:,1:].values);
y_final = le.inverse_transform( clf.predict(xg_test).reshape([62096,]).astype(int) );
y_final = pd.DataFrame(y_final);
numbahs = testing['id']
df = pd.concat([numbahs,y_final],axis=1)
df.columns = ['id','country']
df.to_csv("xgboost_take7.csv",index=False)

# Save model
clf.save_model('xgb_take7.model');

