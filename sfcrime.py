'''
basic naive_bayes model 
for predicting the crime based on various contributing factors 
on a test file.
estimated log_loss on test_train_splits --> 2.5917
logg_loss on submission file(as per kaggle submission) --> 2.58
'''

#load all requisite libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler 
from sklearn.naive_bayes import BernoulliNB

#load data
train=pd.read_csv('train.csv',parse_dates=['Dates'])
test=pd.read_csv('test.csv',parse_dates=['Dates'])

#some visuaizations
groups = train.groupby("Category")["Category"].count()
groups = groups.sort_values(ascending=1)
plt.figure()
groups.plot(kind='bar', title="Crime/category")
#print(groups)

groups = train.groupby("PdDistrict")["PdDistrict"].count()
groups = groups.sort_values(ascending=1)
plt.figure()
groups.plot(kind='bar', title="Crime/city")

groups = train.groupby("DayOfWeek")["DayOfWeek"].count()
groups = groups.sort_values(ascending=1)
plt.figure()
groups.plot(kind='bar', title="Crime/Day")
plt.show()


#function to bring date format in numerical form 
def timesplit(frame):
	frame['Hour']= frame.Dates.dt.hour
	frame['Month']= frame.Dates.dt.month
	frame['Year']= frame.Dates.dt.year

timesplit(train)

#create dummies for Day District Hours and Years from training data(train.csv)
dumies=pd.get_dummies(train['DayOfWeek'])
dumies2=pd.get_dummies(train['PdDistrict'])
dumies3=pd.get_dummies(train['Hour'])
dumies4=pd.get_dummies(train['Year'])
train=train.join(dumies).join(dumies2)

#creating the train data
X=train[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','BAYVIEW','CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK','RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN','X','Y']]
X=X.join(dumies3).join(dumies4)

#creating test data 
y=train[['Category']]
le=LabelEncoder().fit(train.Category)
crime=le.transform(train.Category)
y.pop('Category')
y['Category']=crime

timesplit(test)

#creating dummies for the test.csv similarly
dumies=pd.get_dummies(test['DayOfWeek'])
dumies2=pd.get_dummies(test['PdDistrict'])
dumies3=pd.get_dummies(test['Hour'])
dumies4=pd.get_dummies(test['Year'])
test=test.join(dumies).join(dumies2)
#creating Id column for submission file
test['Id']=xrange(884262);ids=test.Id

#creating test data for submission purposes
X_test2=test[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','BAYVIEW','CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK','RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN','X','Y']]
X_test2=X_test2.join(dumies3).join(dumies4)

#creating testing and training data for analysing log_loss
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=43,test_size=0.2)

#feature scaling(for the longitude and latitude)
scaler=StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test2=scaler.transform(X_test2)

#using naive_bayes classifier
clf=BernoulliNB()

#fitting training data on the classifier 
clf.fit(X_train,y_train)

#calculating estimated log_loss
y_pred=clf.predict_proba(X_test)

print (log_loss(y_test,y_pred))

'''
# creating submission file 
submission=pd.DataFrame(y_pred,index=ids,columns=le.classes_)
submission.to_csv('submission.csv')
'''