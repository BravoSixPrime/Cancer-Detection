
import pandas as pd 
import numpy as np

#Spliting dataset into dependent and independent variables
data = pd.read_csv('wdbc.data')
X = data.iloc[:,3:]
y = data.iloc[:,1]

# Encoding the data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(y)

#Splitting datassts into traing and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Applying Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Creating a model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0,degree=7)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Calculating the accuracy
from sklearn import metrics
acc = metrics.accuracy_score(y_test, y_pred)

print("Model Accuracy: "+str(acc*100)+" percent")

"""Accuracy found with different models:
SVM linear:97.37
Naive Bayes: 89.47
Decision Tree: 91.23
Random Forest:96.49
SVM kernel: 98.25
KNN: 96.49
"""
