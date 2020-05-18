#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[2]:


exceldata = pd.read_excel("Downloads\COVID-19.xlsx")


# In[3]:


exceldata.head()


# In[4]:


X=exceldata[['Sno','age','gender','body temperature','Dry Cough','sour throat','weakness','breathing problem','drowsiness','pain in chest'
            ,'pain in chest','travel history to infected countries','diabetes','heart disease','lung disease','stroke or reduced immunity'
            ,'stroke or reduced immunity','symptoms progressed','high blood pressue','kidney disease','change in appetide','Loss of sense of smell']]


# In[5]:


Y=exceldata[['Corona result']]


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)


# In[8]:


X_train


# In[9]:


len(X_train)


# In[10]:


len(X_test)


# In[11]:


X_test


# In[12]:


from sklearn.linear_model import LogisticRegression


# In[13]:


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, Y_train)


# In[14]:


clf.predict(X_test)


# In[15]:


Y_test


# In[16]:


clf.score(X_test,Y_test)


# In[17]:


pred =clf.predict(X_test)
clf.score(X_train,Y_train)
clf.score(X_test,Y_test)


# In[18]:


print(accuracy_score(pred, Y_test))
print(confusion_matrix(pred, Y_test))


# In[19]:


import pickle


# In[20]:


with open('model_pkl','wb') as f:
    pickle.dump(clf,f)


# In[21]:


with open('model_pkl','rb') as f:
    mp = pickle.load(f)


# In[22]:


mp.predict(X_test)


# In[23]:


X_test.shape


# In[26]:


prediction=mp.predict([[1,20,1,98.6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])


# In[27]:


prediction


# In[ ]:




