import sys
import numpy as np 
import pandas as pd 
from keras.models import Sequential
import tensorflow as tf 
from keras.utils import np_utils
from keras.layers.core import Dense, Activation

x=pd.read_csv('student_data.csv')
print(x)
from sklearn.preprocessing import OneHotEncoder
'''since we have 4 ranks 1,2,3,4 we will have to one hot encode them'''
l1=[]
l2=[]
l3=[]
l4=[]
i=0
y=x['admit']
print(y)
y=np.array(y)
y=y.reshape(-1,1)
onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(y)
i=0
while i<len(x['rank']):
      if x['rank'][i]==1:
        l1.append(1)
      else:
         l1.append(0)  	
      if x['rank'][i]==2:
        l2.append(1)
      else:
         l2.append(0)  	
      if x['rank'][i]==3:
        l3.append(1)
      else:
         l3.append(0)  	
      if x['rank'][i]==4:
        l4.append(1)
      else:
         l4.append(0)
      i=i+1     	 
input=[l1,l2,l3,l4,x['gre'][0:],x['gpa'][0:]]

input=np.array(input)
input=np.transpose(input)
''' we have nit normalised the data/ just trying to implement the model without it'''
fnn = Sequential()
fnn.add(Dense(256))
fnn.add(Activation('sigmoid'))
fnn.add(Dense(32))
fnn.add(Activation('sigmoid'))
fnn.add(Dense(2))
fnn.add(Activation('sigmoid'))
fnn.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fnn.fit(input,y,epochs=1000)
score=fnn.evaluate(input,y)