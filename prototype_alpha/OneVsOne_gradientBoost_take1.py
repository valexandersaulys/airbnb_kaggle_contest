import pandas as pd
from sklearn.cross_validation import train_test_split

training = pd.read_csv("protoAlpha_training.csv")
testing = pd.read_csv("protoAlpha_testing.csv")

X = training.iloc[:,1:-1].values    # transform to NumPy array b/c of some...
                                    # IndexError: indices are out-of-bounds
y = training['country_destination'].values
x_train,x_valid,y_train,y_valid = train_test_split(X,y,test_size=0.3,random_state=None)


# Train classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

clf = OneVsOneClassifier(GradientBoostingClassifier(n_estimators=1000,verbose=100))
clf.fit(x_train,y_train)


# Run Predictions
from sklearn.metrics import confusion_matrix, accuracy_score
y_preds = clf.predict(x_valid)
print( confusion_matrix(y_valid,y_preds) );
print( "Accuracy: %f" % (accuracy_score(y_valid,y_preds)) );
f = open('OneVsOne_gradientBoost_take1.txt', 'w')
f.write( str(confusion_matrix(y_valid,y_preds)) );
f.write( "\nAccuracy: %f" % (accuracy_score(y_valid,y_preds)) );
f.write( "\nOneVsOneClassifier(...)" );
f.write( "\nclf = GradientBoostingClassifier(n_estimators=30,verbose=100)" );


# Now on to final submission
y_final = pd.DataFrame(clf.predict(testing.iloc[:,1:]).reshape([62096,]));
numbahs = testing['id']
df = pd.concat([numbahs,y_final],axis=1)
df.columns = ['id','country']
df.to_csv("OneVsOne_gradientBoost_take1.csv",index=False)
