import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


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
# print("Train set size:", len(train_df))
# print("Validation set size:", len(valid_df))
# print("Test set size:", len(test_df))

# # Count the occurrences of each class in the training set
# class_counts = train_df['class'].value_counts()
# print("Number of objects with class 1 in the training set:", class_counts[1])
# print("Number of objects with class 0 in the training set:", class_counts[0])

##### <<Scaling a Dataset/Under/Oversampling>> #####
selected_features = ["fAsym", "fLength", "fM3Long", "fAlpha"]

def sampling(dataframe, features, oversample=False, undersample=False):
   features_data = dataframe[features].values
   label = dataframe["class"].values
   
   scaler = StandardScaler()
   features_data = scaler.fit_transform(features_data)
   
   if undersample:
    rus = RandomUnderSampler()
    features_data, label = rus.fit_resample(features_data,label)

   if oversample:
    ros = RandomOverSampler()
    features_data, label = ros.fit_resample(features_data,label)

    
   return features_data, label

def scale_dataset(dataframe, features):
   
   features_data = dataframe[features].values
   label = dataframe["class"].values
   
   scaler = StandardScaler()
   features_data = scaler.fit_transform(features_data)
   data = np.hstack((features_data, np.reshape(label, (-1,1))))
   
   return data
  

features_train_oversample, label_train_oversample = sampling(train_df, selected_features,oversample=True, undersample=False)
features_train_undersample, label_train_undersample = sampling(train_df, selected_features,oversample=False, undersample=True)
features_valid, label_valid = sampling(valid_df, selected_features, oversample=False, undersample=False)
features_test, label_test = sampling(test_df, selected_features,oversample=False, undersample=False)

train = scale_dataset(train_df,selected_features)
valid = scale_dataset(valid_df,selected_features)
test = scale_dataset(test_df,selected_features)

# print(features_train_oversample)
# print("This is the label_train oversample:", len(label_train_oversample))
# print("This is the label_train UnderSample:", len(label_train_undersample))


# print("1 in labe_oversample: ", sum(label_train_oversample == 1))
# print("0 in labe_oversample: ", sum(label_train_oversample == 0))


# print("1 in label_undersample: ", sum(label_train_undersample == 1))
# print("0 in label_undersample: ", sum(label_train_undersample == 0))

#this is the main branch
