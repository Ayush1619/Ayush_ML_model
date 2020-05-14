#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[2]:


exceldata = pd.read_excel("C:\\Users\\Ayush\\Downloads\\covid19-symptoms-dataset\covid19-symptoms-dataset.xlsx")


# In[3]:


exceldata.head()


# In[6]:


X=exceldata[['Dry Cough','High Fever','Sore Throat','Difficulty in breathing']]


# In[7]:


Y=exceldata[['Infected with Covid19']]


# In[4]:


from sklearn.model_selection import train_test_split


# In[277]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)


# In[278]:


X_train


# In[279]:


len(X_train)


# In[280]:


len(X_test)


# In[281]:


X_test


# In[282]:


from sklearn.linear_model import LogisticRegression


# In[283]:


clf = LogisticRegression()
clf.fit(X_train, Y_train)


# In[284]:


clf.predict(X_test)


# In[285]:


Y_test


# In[286]:


clf.score(X_test,Y_test)


# In[287]:


pred =clf.predict(X_test)
clf.score(X_train,Y_train)
clf.score(X_test,Y_test)


# In[288]:


print(accuracy_score(pred, Y_test))
print(confusion_matrix(pred, Y_test))


# In[289]:


import pickle


# In[290]:


with open('model_pickle','wb') as f:
    pickle.dump(clf,f)


# In[291]:


with open('model_pickle','rb') as f:
    mp = pickle.load(f)


# In[292]:


mp.predict(X_test)


# In[293]:


X_test.shape


# In[294]:


prediction=mp.predict([[15,17,0,18]])


# In[295]:


prediction


# In[296]:


prediction=mp.predict([[17,17,0,0]])


# In[297]:


prediction


# In[298]:


prediction=mp.predict([[0,8,0,0]])


# In[299]:


prediction


# In[300]:


prediction=mp.predict([[2,5,0,0]])


# In[301]:


prediction


# In[302]:


prediction=mp.predict([[2,4,9,12]])


# In[303]:


prediction


# In[304]:


prediction=mp.predict([[5,0,0,4]])


# In[305]:


prediction


# In[ ]:




