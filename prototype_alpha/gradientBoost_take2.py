"""
Take 2 on the GradientBoost, predicting for country_destinations.

Use labels in confusion_matrix(y_true,y_preds,labels=[]) to order 
the labels in the confusion matrix to see whats overrepresented in the 
target files for the Airbnb contest.
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

'NDF' is over-represented
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

training = pd.read_csv("protoAlpha_training.csv")
testing = pd.read_csv("protoAlpha_testing.csv")

X = training.iloc[:,1:-1]
y = training['country_destination']

x_train,x_valid,y_train,y_valid = train_test_split(X,y,test_size=0.3,random_state=None)

labels_order = np.unique(y_train.values)

# Train classifier
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=10,verbose=100)
clf.fit(x_train,y_train)

# Run Predictions
from sklearn.metrics import confusion_matrix, accuracy_score
y_preds = clf.predict(x_valid)
yt_preds = clf.predict(x_train)

# Print Predictions
print(labels_order);
print( confusion_matrix(y_train,yt_preds,labels=labels_order) );
print( confusion_matrix(y_valid,y_preds,labels=labels_order) );
print( "Accuracy: %f" % (accuracy_score(y_valid,y_preds)) );

# Save metrics to text file
f = open('gradientBoost_take2.txt', 'w')
f.write( str(labels_order) );
f.write( str(confusion_matrix(y_valid,y_preds,labels=labels_order)) );
f.write( "\nAccuracy: %f" % (accuracy_score(y_valid,y_preds)) );
f.write( "\nclf = GradientBoostingClassifier(n_estimators=70,verbose=100)" );

# Now on to final submission
y_final = pd.DataFrame(clf.predict(testing.iloc[:,1:]).reshape([62096,]));
numbahs = testing['id']
df = pd.concat([numbahs,y_final],axis=1)
df.columns = ['id','country']
df.to_csv("gradientBoost_take2.csv",index=False)
