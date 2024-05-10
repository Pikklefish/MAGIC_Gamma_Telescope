import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

#### <Bock,R.. (2007). MAGIC Gamma Telescope. UCI Machine Learning Repository. https://doi.org/10.24432/C52C8B.> #####


##### <<Set up data coloumns titles (features) and read in csv file>> #####
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df.head()
# print(df.head())

##### <<Compares 'class' coloumn (label) with string "g" and returns boolean (true or false). astype(int) converts it to (true=1, false=0)>>####
df["class"]=(df["class"] == "g").astype(int)
# print(df.head())

##### <<For data analysis we create a historgram/print standard deviation/print data count>> ####
# for label in cols[:-1]:
#    plt.figure()
#    plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.7, density=True)
#    plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.7, density=True)
#    plt.title=label
#    plt.ylabel("Probability")
#    plt.xlabel(label)
#    plt.legend()
#    plt.savefig(f"{label}.png")
# plt.show()
# std_devs = df.groupby('class').std()
# print(std_devs)
# class_counts = df['class'].value_counts()
# print(class_counts)


##### <<Create Train Valid and test data set>> #####
train, valid, test = np.split(df.sample(frac=1),[int(0.6*len(df)), int(0.8*len(df))])
# print("Train set size:", len(train))
# print("Validation set size:", len(valid))
# print("Test set size:", len(test))

