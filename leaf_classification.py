'''
Using basic logistic regression
with some tuning
i ahcieved top 100 rank  :)
Abhishek Bhatt(Jamia Millia Islamia B.Tech (CSE))
'''
#import required libraries
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#read files from pwd
train=pd.read_csv('train.csv')											
test=pd.read_csv('test.csv')

# preparing y_train by encoding the non-numerical values
le=LabelEncoder().fit(train.species)
y_train=le.transform(train.species)   									

#dropping the id column as it is useless for training 
X_train=train.drop(['id','species'],axis=1)

#feature scaling(very important step)
scaler=StandardScaler().fit(X_train) 									
X_train=scaler.transform(X_train)     								

#training data by using exhaustive search for optimal values of C->a Logistic Regression parameter(C=1/lambda)

#various C values to choose from for exhaustive search
C_vals={'C':[0.0001,0.001,0.01,0.1,1,10,100,200,500,1000]}  		
logreg=LogisticRegression(solver='lbfgs', multi_class='multinomial')   
grid=GridSearchCV(logreg,C_vals,cv=10,scoring='log_loss')				

#fitting data for training
grid.fit(X_train,y_train)												

#creating id column for submision file
ids=test.id  															

#creating test file for calculating probablity atrix for each class(species)
X_test=test.drop(['id'],axis=1)											
X_test=scaler.transform(X_test)		

#create predictive probablities for each specie of leaf
y_test=grid.predict_proba(X_test)										

#creating submission file 
submission=pd.DataFrame(y_test,index=ids,columns=le.classes_)			
submission.to_csv('submission.csv')										
