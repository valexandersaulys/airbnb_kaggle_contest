"""
Dimenionsality Reduction with PCA
"""
from sklearn.decomposition import PCA
import pandas as pd

training = pd.read_csv("protoAlpha_training.csv")
testing = pd.read_csv("protoAlpha_testing.csv")

one = PCA(n_components=2).fit_transform(training.iloc[:,1:-1]);
traf = pd.DataFrame(one);
df = pd.concat([training.iloc[:,0], traf, training.iloc[:,-1]],axis=1);

two = PCA(n_components=2).fit_transform(testing.iloc[:,1:]);
tesf = pd.DataFrame(two);
tf = pd.concat([testing.iloc[:,0], tesf],axis=1);

df.to_csv("protoBeta_training.csv",index=False);
tf.to_csv("protoBeta_testing.csv",index=False);

