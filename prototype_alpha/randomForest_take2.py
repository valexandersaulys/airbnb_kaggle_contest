"""
Take 1 on the RandomForest, predicting for country_destinations.
"""
import pandas as pd
from sklearn.cross_validation import train_test_split

training = pd.read_csv("protoAlpha_training.csv")
testing = pd.read_csv("protoAlpha_testing.csv")

X = training.iloc[:,1:-1]
y = training['country_destination']



x_train,x_valid,y_train,y_valid = train_test_split(X,y,test_size=0.3,random_state=None)

# Train classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=50,n_jobs=5))
clf.fit(x_train,y_train)
print( clf.feature_importances_ );

# Run Predictions
from sklearn.metrics import confusion_matrix, accuracy_score
y_preds = clf.predict(x_valid)
print( confusion_matrix(y_valid,y_preds) );
print( "Accuracy: %f" % (accuracy_score(y_valid,y_preds)) );
f = open('randomForest_take2.txt', 'w')
f.write( str(confusion_matrix(y_valid,y_preds)) );
f.write( "\nAccuracy: %f" % (accuracy_score(y_valid,y_preds)) );
f.write( "\nOneVsOneClassifier()" );
f.write( "\nclf = RandomForestClassifier(n_estimators=1000)" );

# Now on to final submission
y_final = pd.DataFrame(clf.predict(testing.iloc[:,1:]).reshape([62096,]));
numbahs = testing['id']
df = pd.concat([numbahs,y_final],axis=1)
df.columns = ['id','country']
df.to_csv("randomForest_take2.csv",index=False)
