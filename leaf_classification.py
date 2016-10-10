import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


train=pd.read_csv('leaf.csv')											#read files
test=pd.read_csv('test.csv')

le=LabelEncoder().fit(train.species)
y_train=le.transform(train.species)   									#yaha species k naam ko numbrs me encode kia

X_train=train.drop(['id','species'],axis=1)

scaler=StandardScaler().fit(X_train) 									#feature scaling krne k lie h 
X_train=scaler.transform(X_train)     									#m to har baar lagaunga chahe zrurat ho ya na ho



C_vals={'C':[0.0001,0.001,0.01,0.1,1,10,100,200,500,1000]}  			#ek parameter ki dictionay bnai taki i can loop over multiple values jisme se best utha sku for my model
logreg=LogisticRegression(solver='lbfgs', multi_class='multinomial')   	#solver to algo h zada pta nhi but advanced h islie use kia(and multiclass k lie use hota h itna pata h )....multinomial (one vs all implement krne k lie)   random 
grid=GridSearchCV(logreg,C_vals,cv=10,scoring='log_loss')				#exhaustive method h jo har combo ka X_train,y_train use krke best c_val ki value k lie min log_loss dega fir uspe data fit krega(efficient method hota h )
grid.fit(X_train,y_train)												#data fit krdo

ids=test.id  															#id ko variable me store krlo
X_test=test.drop(['id'],axis=1)											#id remove krdo qki yaha zrurat nhi	
X_test=scaler.transform(X_test)											#test input ki feature scaling
y_test=grid.predict_proba(X_test)										# ab har class ki probablity jo predict hogi use store krlo
submission=pd.DataFrame(y_test,index=ids,columns=le.classes_)			#ek submission naam ke dataframe me table store krlo with all species as column names,ids as indexes, and y_test as their respective probablities
submission.to_csv('submission.csv')										#csv bna lo
