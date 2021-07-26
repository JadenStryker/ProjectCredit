#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# In[28]:


df = pd.read_csv('creditcard.csv')
df.describe()


# In[11]:


df[df['Class'] == 0].shape


# In[25]:


#Here I can see that there are 25 variable values as well as time, amount, and class
#The dataset doesnt offer any names for the variables for privacy reasons
#492 fraud, 284315 not fraud. Imbalanced class problem? Lets take a look at the mean and median values of amount
print("percent fraud", 495/284315)


# In[24]:


dfFraud = df[df['Class'] == 1]
print(f"Fraud amount mean {dfFraud['Amount'].mean()}")

dfNF = df[df['Class'] == 0]
print(f"Not Fraud amount mean {dfNF['Amount'].mean()}")


# In[26]:


#We can see that fradulant charges have a much higher mean. STD of about 250 as well. Lets check out distributions of timeand amount 
#or only two known features


# In[41]:


fig, ax = plt.subplots(1, 2, figsize=(16,4))

amount = df['Amount'].values
time = df['Time'].values

sns.distplot(amount, ax=ax[0], color='y')
ax[0].set_title('Distribution of Transaction Amount', fontsize=16)

sns.distplot(time, ax=ax[1], color='r')
ax[1].set_title('Distribution of Transaction Time', fontsize=16)


plt.show()


# In[43]:


#Here we can see the distributions of transaction amount to be close to zero and the distribution of time to be 
#around 50000 seconds and 15000 seconds. Lets check for the positive fraud values
#There is also a scaling issue that is very clear on the Transaction amount part


# In[45]:


fig, ax = plt.subplots(1, 2, figsize=(16,4))

amount = dfFraud['Amount'].values
time = dfFraud['Time'].values

sns.distplot(amount, ax=ax[0], color='y')
ax[0].set_title('Distribution of Transaction Amount', fontsize=16)

sns.distplot(time, ax=ax[1], color='r')
ax[1].set_title('Distribution of Transaction Time', fontsize=16)


plt.show()


# In[46]:


#We can see that the values are generally the same. with both curves following the overall shape
#As mentioned there is a class imbalance problem so next I will create a sub sample to create a 50/50 split. 
#If this wasnt adressed the classifier could simply classify all values as not fraud and have an accuracy of 99.7%


# In[54]:


# Using a standard scalar to scale down the distribution values

std_scalar = StandardScaler()

df['time_scale'] = std_scalar.fit_transform(df['Time'].values.reshape(-1,1))
df['amount_scale'] = std_scalar.fit_transform(df['Amount'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)


# In[59]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

X_train, X_test, y_train, y_test = train_test_split(df, df['Class'], test_size=0.33, random_state=42)


# In[65]:


#I will use a technique called random under sampling to help deal with the imbalanced class problem


# In[78]:


dfNotFraud = dfNF[:492]
even_df = pd.concat([dfFraud, dfNotFraud])
even_df.shape
even_df[even_df['Class'] == 0].shape
#Both are even. This isnt ideal because I just chose the first few values and it didnt sample randomly. I will see if
#there are better techniques going forward. 

