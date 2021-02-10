#!/usr/bin/env python
# coding: utf-8

# PCALog: Binary Classification of the OI Dataset
# 
# The dataset consist of information on the patient's age, sex, OI type and the landmark coordinates.


import sklearn, csv
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load


# Loading and Preprocessing
# --------------------------------------------------

# reading the csv file to a pandas dataframe
raw_df = pd.read_csv('oi_ml_db.csv')
oi_df = raw_df.drop(['Unnamed: 0', 'Genetic'], axis=1)
pd.DataFrame.head(oi_df)

# change the control group from 'c' to '0', '3' to '1' and '4' to '1'
oi_df = oi_df.replace({'Group':'c'}, 0)
oi_df = oi_df.replace({'Group':'3'}, 1)
oi_df = oi_df.replace({'Group':'4'}, 1)
# change M to 0 and F to 1
oi_df = oi_df.replace({'Gender':'M'}, 0)
oi_df = oi_df.replace({'Gender':'F'}, 1)
# we do not need the file.no anymore
oi_df = oi_df.drop(['File.No.'], axis=1)
# normalization of age
# oi_df = (oi_df-oi_df.mean())/oi_df.std()
pd.DataFrame.head(oi_df)


# shuffle the dataframe
rand_df = sklearn.utils.shuffle(oi_df)

# create the complete y and X sets from the shuffled data and scale the X values
X_full = rand_df.drop(['Group'], axis=1)
y_full = rand_df['Group']
X_full = preprocessing.scale(X_full)

# splitting X and y for train and test 80/20
X_sets = np.split(X_full, [245, 61])
X_train = X_full[0:245,]
X_test = X_full[245:306,]
y_train = y_full.values[0:245,]
y_test = y_full.values[245:306,]

# cast y values as integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print('train set: ' + str(X_train.shape) + '\ntest set: ' + str(X_test.shape))

# PCALog model implementationa and trainig
# --------------------------------------------------

# PCA for dimensional reduction of our data 138 features to 20 features
pca = PCA(n_components=20)
pComp = pca.fit(X_train)
X_train_pca = pComp.transform(X_train)
X_test_pca = pComp.transform(X_test)

# Logistic regression
lr_clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train_pca, y_train)
lr_clf.score(X_test_pca, y_test)

# logistic regression performed a solid 98.36% on the test set just like the svm
# 20 compoment PCA will give us 100% score on our training set

# model metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# Visualizations
# --------------------------------------------------

x0 = None
x1 = None

def split(y, array):
    g0 = []
    g1 = []
    for i in range(len(y)):
        if y[i] == 0:
            g0.append(array[i])
        if y[i] == 1:
            g1.append(array[i])
    return np.array(g0), np.array(g1)

# logistic function P(X) = e^{b0+b1X}/1+e^{b0+b1X} schematic
def log_func(X):
    e = np.exp(lr_clf.coef_*X)
    y = e / (1+e)
    return y

f = 1

xy = np.concatenate((X_test_pca[:,f], log_func(X_test_pca)[:,f]))
xy = np.reshape(xy, (2,61))
xy = xy.T
xy = np.sort(xy, axis=0)
px = xy[:,0]
py = xy[:,1]

x0, x1 = split(y_test, X_test_pca)
y0, y1 = split(y_test, X_test_pca)
plt.scatter(x0[:,f], lr_clf.predict(x0), label = 'Controls')
plt.scatter(x1[:,f], lr_clf.predict(x1), label = 'OI Subjects')
#plt.scatter(X_test[:,f], log_func(X_test)[:,f], label = 'Logistic Function')
plt.plot(px, py, label = 'Logistic Function')
plt.ylabel('Model Prediction')
plt.xlabel('Principal Component 1')
plt.axhline(.5, color='.5')
plt.axhline(0, color='.5')
plt.axhline(1, color='.5')
plt.legend()
plt.savefig('log_func.jpg')
plt.show()


# confusion matrix
y_pred = lr_clf.predict(X_test_pca)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.title("Confusion Matrix Binary Classification, Control - Affected", y=1.08)
plt.tight_layout()
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('conf_mat_bin.png')
plt.show()


# save the model with joblib
dump(lr_clf, 'bin_clf_ca.joblib') 

print(metrics.classification_report(y_test, y_pred))

y_pred_proba = lr_clf.predict_proba(X_test_pca)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Logistic Regression Model, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.savefig('log_roc.jpg')
plt.show()
