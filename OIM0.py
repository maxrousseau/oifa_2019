#!/usr/bin/env python
# coding: utf-8

# # Modified Type Classification of the OI Dataset
# 
# The dataset consist of information on the patient's age, sex, OI type and the landmark coordinates.
# 
# In this notebook we will be experimenting with various algorithms. Before we start we will need to prepare our data.
# 
# Patients will be grouped as follows 0 = control, 1 = types 1 and 4, 3 = type 3

# In[2]:


import sklearn, csv
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load


# In[3]:


# reading the csv file to a pandas dataframe
raw_df = pd.read_csv('oi_ml_db.csv')
oi_df = raw_df.drop(['Unnamed: 0', 'Genetic'], axis=1)
pd.DataFrame.head(oi_df)


# In[4]:


# change the control group from 'c' to '0', '3' to '1' and '4' to '1'
oi_df = oi_df.replace({'Group':'c'}, 0)
oi_df = oi_df.replace({'Group':'4'}, 1)
# change M to 0 and F to 1
oi_df = oi_df.replace({'Gender':'M'}, 0)
oi_df = oi_df.replace({'Gender':'F'}, 1)
# we do not need the file.no anymore
oi_df = oi_df.drop(['File.No.'], axis=1)
# normalization of age
# oi_df = (oi_df-oi_df.mean())/oi_df.std()
pd.DataFrame.head(oi_df)


# In[5]:


# shuffle the dataframe
rand_df = sklearn.utils.shuffle(oi_df)

# create the complete y and X sets from the shuffled data and scale the X values
X_full = rand_df.drop(['Group'], axis=1)
y_full = rand_df['Group']
X_full = preprocessing.scale(X_full)

# PCA for dimensional reduction of our data 138 features to 10 features
pca = PCA(n_components=30)
pComp = pca.fit_transform(X_full)

# splitting X and y for train and test 80/20
X_sets = np.split(pComp, [245, 61])
X_train = pComp[0:245,]
X_test = pComp[245:306,]
y_train = y_full.values[0:245,]
y_test = y_full.values[245:306,]

# cast y values as integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print('train set: ' + str(X_train.shape) + '\ntest set: ' + str(X_test.shape))


# In[6]:


# Multi Layer Perceptron (AKA: shallow neural network)
# more sofisticated classifier but may tend to overfit
mlp = MLPClassifier(hidden_layer_sizes=(30,), max_iter=10, alpha=1e-4, solver='sgd', 
                    verbose=10, tol=1e-4, random_state=1, learning_rate_init=.1)
mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

# modified type classification seems to be working very well for this dataset with a PCA of 30 components


# In[7]:


# Linear Support Vector Machine
svm_clf = LinearSVC(random_state=0, tol=1e-5)
svm_clf.fit(X_train, y_train)
pred = svm_clf.predict(X_test)
right = 0
wrong = 0
index = 0
for p in pred:
    if p == y_test[index]:
        right += 1
    if pred[index] != y_test[index]:
        wrong += 1
    index += 1
print('Test score: ' + str(right/61) + '%')

# the linear support vector machine seems to be performing similarily to the NN 


# In[8]:


# Logistic regression
lr_clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
lr_clf.score(X_test, y_test)

# great performance with PCA of 30


# In[10]:


# confusion matrix
y_pred = lr_clf.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.title("Confusion Matrix Modified Classification", y=1.08)
plt.tight_layout()
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('conf_mat_mod.png')
plt.show()


# In[11]:


# save the model with joblib
dump(lr_clf, 'mod_clf.joblib') 


# In[13]:


print(metrics.classification_report(y_test, y_pred))


# In[ ]:




