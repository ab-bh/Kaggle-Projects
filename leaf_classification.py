import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


train=pd.read_csv('leaf.csv')											#read files
test=pd.read_csv('test.csv')

le=LabelEncoder().fit(train.species)
y_train=le.transform(train.species)   									

X_train=train.drop(['id','species'],axis=1)

scaler=StandardScaler().fit(X_train) 									
X_train=scaler.transform(X_train)     								



C_vals={'C':[0.0001,0.001,0.01,0.1,1,10,100,200,500,1000]}  		
logreg=LogisticRegression(solver='lbfgs', multi_class='multinomial')   
grid=GridSearchCV(logreg,C_vals,cv=10,scoring='log_loss')				
grid.fit(X_train,y_train)												

ids=test.id  															
X_test=test.drop(['id'],axis=1)											
X_test=scaler.transform(X_test)											
y_test=grid.predict_proba(X_test)										
submission=pd.DataFrame(y_test,index=ids,columns=le.classes_)			
submission.to_csv('submission.csv')										
