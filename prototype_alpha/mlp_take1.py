"""
First attempt at neural network. 

classes for target variables is done via pd.get_dummies()
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

training = pd.read_csv("protoAlpha_training.csv")
testing = pd.read_csv("protoAlpha_testing.csv")

X = training.iloc[:,1:-1]
y = training['country_destination']

x_train,x_valid,y_train,y_valid = train_test_split(X,y,test_size=0.3,random_state=None)

# Train classifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adagrad, SGD

classes = len(np.unique(y_train.values).tolist()); # Get the number of classes
y_train = pd.get_dummies(y_train,dummy_na=False); # convert with pd.get_dummies()

clf = Sequential()
clf.add( Dense(10000, input_dim=x_train.shape[1], init='uniform') )
clf.add( Activation('sigmoid') );  #'sigmoid', 'tanh' too
clf.add( Dense(classes) );  # I think its one per country
clf.add( Activation('softmax') );
#sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
agd = Adagrad(lr=0.01, epsilon=1e-6);
clf.compile(loss="categorical_crossentropy",optimizer=agd);
clf.fit(x_train.values,y_train.values,batch_size=128,nb_epoch=100)


# Run Predictions
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

y_p = clf.predict(x_valid.values,batch_size=128,verbose=1)
y_preds = pd.DataFrame(y_p,columns=y_train.columns).idxmax(axis=1)

print( confusion_matrix(y_valid,y_preds) );
print( "Accuracy: %f" % (accuracy_score(y_valid,y_preds)) );
print( "Precision: %f" % (precision_score(y_valid,y_preds)) );
print( "Recall: %f" % (recall_score(y_valid,y_preds)) );

f = open('mlp_take1.txt', 'w')
f.write( str(confusion_matrix(y_valid,y_preds)) );
f.write( "\nAccuracy: %f" % (accuracy_score(y_valid,y_preds)) );
f.write( "\nPrecision: %f" % (precision_score(y_valid,y_preds)) );
f.write( "\nRecall: %f" % (recall_score(y_valid,y_preds)) );
f.write( "\nagd = Adagrad(lr=0.01, epsilon=1e-6)" );
f.write( "\nclf.fit(x_train.values,y_train.values,batch_size=1,nb_epoch=25)" );


# Now on to final submission
y_f = pd.DataFrame(clf.predict(testing.iloc[:,1:].values,batch_size=128),columns=y_train.columns);
y_final = y_f.idxmax(axis=1);
numbahs = testing['id']
df = pd.concat([numbahs,y_final],axis=1)
df.columns = ['id','country']
df.to_csv("mlp_take1.csv",index=False)
