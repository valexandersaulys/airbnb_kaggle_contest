"""
I had trouble with this.

Take 4 on the Gradient Boost, predicting for country_destinations.
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split

# ------------- Build create_submission function
def create_submission_format(df,dummies):
    # Columns --> ['id','chk_NDF',dummies]
    # dummies will be 'y_data' array (see below)
    # 0 is 'id'
    # 1 is 'chk_NDF'
    # 2: are the dummies
    # array[columns][rows]
    appendage = []; 
    for i in range(len(df)):
        if df[i][1] == 1:
            abc = [ df[i][0], "NDF" ]
        else:
            abc = [ df[i][0], dummies[np.argmax(a[0][2:])] ]
        appendage.append(abc);
    ret_df = pd.DataFrame(appendage,columns=['id','country_destination']);
    return ret_df;
            
# -------------

print("Reading Data");

training = pd.read_csv("protoAlpha_training.csv")
testing = pd.read_csv("protoAlpha_testing.csv")

print("Finished! Now onto creating the 'chk_NDF' column");

# Create 'chk_NDF' column
training['chk_NDF'] = 0;
training['chk_NDF'][ training['country_destination']=='NDF' ] = 1;
training.rename(columns={"other": "Oth"}, inplace=True);  # 'other' later conflicts

print("Split Data");
# Split up data between validation & training
x_data = training.iloc[:,1:108].columns
y_data = pd.get_dummies(training['country_destination']).columns.tolist();
y_data.remove('NDF');  # Don't want this
training = pd.concat( [training,
                       pd.get_dummies(training['country_destination']) ],
                      axis=1);
train, valid = train_test_split(training,test_size=0.4);
submission_columns = ['id','chk_NDF'] + y_data;

print("Create & Fit clf_one");
# Train clf_one data on 'chk_NDF' data
clf_one = GradientBoostingClassifier(n_estimators=50,verbose=1)
clf_one.fit(train[x_data], train['chk_NDF']);
valid_chk_NDF_preds = clf_one.predict(valid[x_data]);

print("Create & Fit clf_two");
# Train clf_two data on 'chk_NDF' data
clf_two = GradientBoostingClassifier(verbose=1)
clf_two.fit(train[x_data][ train['chk_NDF'] == 0 ],
            train["country_destination"][ train['chk_NDF'] == 0 ]);            
            #train[y_data][ train['chk_NDF'] == 0 ]);
valid_dummies_preds = clf_two.predict(valid[x_data]);

print("Get submission format");
# Get Submission Format
pass_submit = [ valid['id'].values.tolist(),
               valid_chk_NDF_preds.tolist(),  # I think its an array
               valid_dummies_preds.tolist() ]
comparison = create_submission_format(pass_submit,y_data);

print("Run Predictions!\n");

# Run Predictions
from sklearn.metrics import confusion_matrix, accuracy_score
print( confusion_matrix(valid['country_destination'],
                        comparison['country_destination']) );
print( "Accuracy: %f" % (accuracy_score(valid['country_destination'],
                                        comparison['country_destination']))  );
f = open('gradientBoost_take1.txt', 'w')
f.write( str(confusion_matrix(valid['country_destination'],
                              comparison['country_destination'])) );
f.write( "\nAccuracy: %f" % (accuracy_score(valid['country_destination'],
                                            comparison['country_destination'])) );
f.write( "\nclf = GradientBoostingClassifier()" );

# Now on to final submission
y_chk_NDF = clf_one.fit(testing[x_data]);
y_cats = clf_two.fit(testing[y_data]);
final_submit = [ testing['id'],
                 y_chk_NDF_preds.tolist(),
                 y_cats.tolist() ]
df = create_submission_format(final_submit,y_data);
df.to_csv("submission_gradientBoost_take4.csv",index=False);
