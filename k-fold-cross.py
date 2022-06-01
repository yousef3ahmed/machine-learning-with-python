import pandas as pd 
data=pd.read_csv('diabetics.csv') #datafram


data.shape
data.head()

X = data.drop('outcome' , 1)
Y = data[ 'outcome' ]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


rf = RandomForestClassifier(n_estimators = 10)

from sklearn.model_selection import KFold

k=5
kfold =KFold(n_splits=k, random_state=None, shuffle=False)

acclist=[]

for train_index, test_index in kfold.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = Y[train_index] , Y[test_index]
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_test, predictions)
    print( matrix )
    print('_________________________________________')
    
    
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(y_test,predictions)
    acclist.append(acc)
    
acc=sum(acclist)/k
print('acc : ',acc)