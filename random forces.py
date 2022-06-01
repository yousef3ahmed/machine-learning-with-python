import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('diabetics.csv')
data.head()

X = data.drop('outcome', 1)
y = data['outcome']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.30)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50)
classifier = classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


result1 = classification_report(y_test, predictions)
print("Classification Report:",result1 )

result = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(result)

result2 = accuracy_score(y_test,predictions)
print("Accuracy:",result2)


from sklearn.metrics import precision_score
pre = precision_score(y_test, predictions)
print("precision = " , pre) 

# import seaborn as sb
# %matplotlib inline
# sb.countplot(x = 'Outcome' , data = data , palette='hls') 
