"""
Take 2 on the GradientBoost, predicting for country_destinations.

Use labels in confusion_matrix(y_true,y_preds,labels=[]) to order 
the labels in the confusion matrix to see whats overrepresented in the 
target files for the Airbnb contest.
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

'NDF' is over-represented, so I'm gonna drop it.
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

# Function to get that shit in order
def get_submission_format(IDs,ndfs,answers):
    # IDs should be dataframe, ndfs & answers can be list/numpyarray
    y_final = [];
    for i in range(len(ndfs)):
        if ndfs[i]==1: # Affirmative
            y_final.append('NDF')
        else:
            y_final.append(answers[i]);
    y_final = pd.DataFrame(y_final);  # Check this, it might need to be reshaped
    df = pd.concat([IDs,y_final],axis=1);
    df.columns = ['id', 'country']
    return df


training = pd.read_csv("protoAlpha_training.csv")
testing = pd.read_csv("protoAlpha_testing.csv")

X = training.iloc[:,1:-1]
y = training['country_destination']

x_train,x_valid,y_train,y_valid = train_test_split(X,y,test_size=0.3,random_state=None)

x_train['NDF'] = 0; x_valid['NDF'] = 0;
x_train['NDF'][ y_train['country_destination']=='NDF' ] = 1;  # idk if this will work
x_valid['NDF'][ y_valid['country_destination']=='NDF' ] = 1;

# Get the 'NDF' column values
yn_train = x_train.iloc[:,:-1]
yn_valid = x_valid.iloc[:,:-1]

labels_order = np.unique(y_train.values)

# Drop the extra columns in x_train
x_train = x_train.iloc[:,:-1]
x_valid = x_valid.iloc[:,:-1]
# First train a classifier for 'NDF' vs everything else
from sklearn.ensemble import GradientBoostingClassifier
clf_one = GradientBoostingClassifier(n_estimators=10,verbose=100)
clf_one.fit(x_train,yn_train);
yn_preds = clf_one.predict(x_valid);
print(  "Accuracy: %f" % accuracy_score(yn_valid,yn_preds) );

# Drop values that are 'NDF' destination
x_t = x_train[ y_train['country_destination'] == 'NDF' ]
x_v = x_valid[ y_valid['country_destination'] == 'NDF' ]
y_t = y_train['country_destination'][y_train['country_destination'] != 'NDF']
y_v = y_valid['country_destination'][y_valid['country_destination'] != 'NDF']

# Next, train a classifier for everything else
clf_two = GradientBoostingClassifier(n_estimators=70,verbose=10)
clf_two.fit(x_t,y_t)
y_p = clf_two.predict(x_v);
print( "Accuracy: %f" % accuracy_score(y_v,y_p) );
"""
# Full run-through for valid
ndf_answers = clf_one.predict(x_valid);
x_vld = x_valid[ndf_answers==1]
y_answers = clf_two.predict(x_vld);
"""
# Get the final testing data answer
x_test = testing.iloc[:,1:];
ndf_test = clf_one.predict(x_test);
x_tst = x_test[ndf_test==1];  # Might need to be a dataframe & not a numpy
y_answers = clf_two.predict(x_tst);

numbahs = testing['id']
df = get_submission_format(numbahs,ndfs=,answers=);
df.to_csv("gradientBoost_take3.csv",index=False)
