#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[3]:


dataset = pd.read_csv("Downloads\symptoms.csv")


# In[4]:


dataset.head()


# In[5]:


X=dataset[['Dry Cough','High Fever','Sore Throat','Difficulty in breathing']]


# In[6]:


Y=dataset[['Infected with Covid19']]


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)


# In[9]:


X_train


# In[10]:


len(X_train)


# In[12]:


len(X_test)


# In[13]:


X_test


# In[14]:


from sklearn.linear_model import LogisticRegression


# In[15]:


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, Y_train)


# In[16]:


clf.predict(X_test)


# In[17]:


Y_test


# In[18]:


clf.score(X_test,Y_test)


# In[19]:


pred =clf.predict(X_test)
clf.score(X_train,Y_train)
clf.score(X_test,Y_test)


# In[20]:


print(accuracy_score(pred, Y_test))
print(confusion_matrix(pred, Y_test))


# In[21]:


import pickle


# In[22]:


with open('model.pkl','wb') as f:
    pickle.dump(clf,f)


# In[23]:


with open('model.pkl','rb') as f:
    mp = pickle.load(f)


# In[24]:


mp.predict(X_test)


# In[25]:


X_test.shape


# In[26]:


prediction=mp.predict([[0,2,3,0]])


# In[27]:


prediction


# In[ ]:




