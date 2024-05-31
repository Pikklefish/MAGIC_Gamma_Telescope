import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

import hashlib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


#### <Bock,R.. (2007). MAGIC Gamma Telescope. UCI Machine Learning Repository. https://doi.org/10.24432/C52C8B.> #####


##### <<Set up data coloumns titles (features) and read in csv file>> #####
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df.head()
# print(df.head())

##### <<Compares 'class' coloumn (label) with string "g" and returns boolean (true or false). astype(int) converts it to (true=1, false=0)>>####
df["class"]=(df["class"] == "g").astype(int)
# print(df.head())




##### <<Create Train Valid and test data set>> #####
train_df, valid_df, test_df = np.split(df.sample(frac=1, random_state=42),[int(0.6*len(df)), int(0.8*len(df))])


##### <<Scaling a Dataset/Under/Oversampling>> #####
def sampling(dataframe, oversample=False, undersample=False):
   features = dataframe[dataframe.columns[:-1]].values
   label = dataframe[dataframe.columns[-1]].values
   
   scaler = StandardScaler()
   features = scaler.fit_transform(features)
   
   if undersample:
    rus = RandomUnderSampler(random_state=42)
    features, label = rus.fit_resample(features,label)

   if oversample:
    ros = RandomOverSampler(random_state=42)
    features, label = ros.fit_resample(features,label)

    
   return features, label

def scale_dataset(dataframe):
   
   features = dataframe[dataframe.columns[:-1]].values
   label = dataframe[dataframe.columns[-1]].values
   
   scaler = StandardScaler()
   features = scaler.fit_transform(features)
   data = np.hstack((features, np.reshape(label, (-1,1))))
   
   return data
  

features_train_oversample, label_train_oversample = sampling(train_df, oversample=True, undersample=False)
features_train_undersample, label_train_undersample = sampling(train_df, oversample=False, undersample=True)
features_valid, label_valid = sampling(valid_df, oversample=False, undersample=False)
features_test, label_test = sampling(test_df, oversample=False, undersample=False)

train = scale_dataset(train_df)
valid = scale_dataset(valid_df)
test = scale_dataset(test_df)



###### <<SVM Model>> #####

svm_model = SVC()

######Oversample
svm_model = svm_model.fit(features_train_oversample, label_train_oversample)

label_pred  = svm_model.predict(features_test)
print("oversample")
print(classification_report(label_test, label_pred))


######Undersample
svm_model = svm_model.fit(features_train_undersample, label_train_undersample)

label_pred  = svm_model.predict(features_test)
print("undersample")
print(classification_report(label_test, label_pred))
#this is the not feature selected branch
