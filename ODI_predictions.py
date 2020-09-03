#!/usr/bin/env python
# coding: utf-8

# ### Importing and preaparing dataset

# In[3]:


import pandas as pd
dataset = pd.read_csv('odi.csv')
X = dataset.iloc[:,[7,8,9,12,13]].values
y = dataset.iloc[:, 14].values


# In[4]:


dataset


# ### Train-Test Split and Scaling Features

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[6]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[7]:


from sklearn.linear_model import LinearRegression #Training model using LinearRegression
lin = LinearRegression()
lin.fit(X_train,y_train)
lin.score(X_test,y_test)


# In[8]:


def custom_accuracy(y_test,y_pred,thresold):
    right = 0
    l = len(y_pred)
    for i in range(0,l):
        if(abs(y_pred[i]-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)


# In[9]:


y_pred = lin.predict(X_test)
score = lin.score(X_test,y_test)*100
print("R-squared value:" , score)
print("Custom accuracy:" , custom_accuracy(y_test,y_pred,20))


# In[10]:


from sklearn.ensemble import RandomForestRegressor #Training Model using RandomForestRegressor
Rf = RandomForestRegressor(n_estimators=100,max_features=None)
Rf.fit(X_train,y_train)


# ### Evaluation

# In[11]:


y_pred = Rf.predict(X_test)
score = Rf.score(X_test,y_test)*100
print("R-squared value:" , score)
print("Custom accuracy:" , custom_accuracy(y_test,y_pred,20))


# ### Prediction on custom input (Current Runs=127, Wickets=1, Overs=15, Striker=78, Non-striker=45)

# In[40]:


import numpy as np
new_prediction = Rf.predict(sc.transform(np.array([[127,1,15,78,45]])))
print("Prediction score:" , new_prediction)


# In[ ]:





# In[ ]:




