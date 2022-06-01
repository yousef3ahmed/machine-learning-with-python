import pandas as pd
import numpy as np
from sklearn import svm

data = pd.read_csv('diabetics.csv')
data.head()


X = data.drop('outcome', 1)
y = data['outcome']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.30)

classifier = svm.SVC(kernel="rbf")
classifier = classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

matrix =  confusion_matrix(y_test, predictions)

print( matrix )


# tn     fp
# fn     tn

from sklearn.metrics import accuracy_score
accc = accuracy_score(y_test, predictions)
print("accuracy = " , accc) 

from sklearn.metrics import precision_score
pre = precision_score(y_test, predictions)
print("precision = " , pre) 