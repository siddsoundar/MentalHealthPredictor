#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

# In[2]:


import numpy as np

# In[3]:


import matplotlib.pyplot as plt

# In[4]:


import seaborn as sns

# In[5]:


from sklearn.model_selection import train_test_split

# In[6]:


from sklearn.linear_model import LinearRegression

# In[23]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# In[25]:


import pickle

# In[26]:


# In[ ]:


# In[8]:


df = pd.read_csv('heart_2020_cleaned.csv')

# In[9]:


sns.scatterplot(x='PhysicalHealth', y='MentalHealth', data=df)
plt.title('PhysicalHealth vs. MentalHealth')
plt.show()

# In[10]:


sns.scatterplot(x='BMI', y='MentalHealth', data=df)
plt.title('BMI vs. MentalHealth')
plt.show()

# In[11]:


sns.scatterplot(x='BMI', y='PhysicalHealth', data=df)
plt.title('BMI vs. PhysicalHealth')
plt.show()

# In[ ]:


# In[12]:


X = df[['PhysicalHealth', 'BMI']]
y = df['MentalHealth']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)

# In[13]:


lr = LinearRegression()
lr.fit(X_train, y_train)

# In[14]:


y_pred = lr.predict(X_test)

# In[15]:


mse = mean_squared_error(y_test, y_pred)

# In[16]:


print("Mean squared error:", mse)

# In[17]:


y = df['MentalHealth']

# In[18]:


print('Mean mental health days:', y.mean())
print('Median mental health days:', y.median())
print('Standard deviation of mental health days:', y.std())

# In[19]:


plt.hist(y, bins=30)
plt.title('Histogram of Mental Health Days')
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.show()

rmse = mean_squared_error(y_test, y_pred, squared=False)

mae = mean_absolute_error(y_test, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)

filename = 'linear_regression_model.pkl'
pickle.dump(lr, open(filename, 'wb'))
