'''

Author: Abhishek Bhatt B.Tech(Jamia Millia Islamia (Vsem))
Using Keras for designing a Neural Net:
3 Layer network:
->1st Layer: 1000 neurons, input weights=192, activation=ReLu, dropout(to tackle overfitting)=0.1
->2nd Layer: 550 neurons, activation=Sigmoid, dropout(to tackle overfitting)=0.2
->3rd Layer: 99 neurons, activation=Softmax

Using below model 
->able to get a best log_loss of 0.014
->Secured a all time top 20 rank (Currently ranked top 30)

'''
#loading all the libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from sklearn.preprocessing import StandardScaler,LabelEncoder
from keras.utils.np_utils import to_categorical as tocat

#initializing random seed value
np.random.seed(10)

#load the train and test data
train=pd.read_csv('leaf.csv')
test=pd.read_csv('test.csv')

#creating y_train for training
le=LabelEncoder().fit(train.species)
y_train=le.transform(train.species)   									

# dropping id column as of no use
X_train=train.drop(['id','species'],axis=1)

# feature scaling
scaler=StandardScaler().fit(X_train) 									
X_train=scaler.transform(X_train)  

#ids for submission file
ids=test.id 

#one hot encoding of y_train values
y_train=tocat(y_train)

#feature scaling for y_test  
X_test=test.drop(['id'],axis=1)											
X_test=scaler.transform(X_test)											

#initiating model
model=Sequential() 

#adding layers

#layer1
model.add(Dense(1000,input_dim=192))
model.add(Activation("relu"))
model.add(Dropout(0.1))

#layer2
model.add(Dense(550))
model.add(Dropout(0.2))
model.add(Activation("sigmoid"))

#layer3
model.add(Dense(99))
model.add(Activation("softmax"))

#compiling model
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#fitting X_train and y_train
params=model.fit(X_train,y_train,nb_epoch=250,batch_size=250,verbose=0)

#creating a multiclass probablistic prediction values
y_test=model.predict_proba(X_test)

#create submission file
submission=pd.DataFrame(y_test,index=ids,columns=le.classes_)
submission.to_csv('submission.csv')