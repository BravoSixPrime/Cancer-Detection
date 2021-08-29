import tensorflow as tf
tf.__version__
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Applying Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Creating a model
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=45, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=45, activation='relu'))
ann.add(tf.keras.layers.Dense(units=45, activation='relu'))
ann.add(tf.keras.layers.Dense(units=45, activation='relu'))
# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 80)

# Part 4 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# Calculating the accuracy
from sklearn import metrics
acc = metrics.accuracy_score(y_test, y_pred)

print("Model Accuracy: "+str(acc*100)+" percent")
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

