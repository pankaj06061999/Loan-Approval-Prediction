#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


# In[7]:


df= pd.read_csv("loan.csv")


# In[8]:


df.head()


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


df['loanAmount_log']=np.log(df['LoanAmount'])
df['loanAmount_log'].hist(bins=20)


# In[12]:


df.isnull().sum()


# In[13]:


df['totalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['totalIncome_log']=np.log(df['totalIncome'])
df['totalIncome_log'].hist(bins=20)


# In[14]:


df.isnull().sum()


# In[15]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)


df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
df.loanAmount_log= df.loanAmount_log.fillna(df.loanAmount_log.mean())


df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)





# In[16]:


df.isnull().sum()


# In[17]:


x= df.iloc[:,np.rP]


# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


print("per of missing gender is %2f%%" % ((df['Gender'].isnull().sum()/df.shape[0])*100))


# In[19]:


print ("number of people who take loan as group by gender:")
print (df['Gender'].value_counts())
sns.countplot(x='Gender',data=df, palette= 'Set1')


# In[20]:


print ("number of people who take loan as group by marital status:")
print (df["Married"].value_counts())
sns.countplot(x='Married',data=df, palette= 'Set1')


# In[21]:


print ("number of people who take loan as group by dependents:")
print (df['Dependents'].value_counts())
sns.countplot(x='Dependents',data=df, palette= 'Set1')


# In[22]:


print ("number of people who take loan as group by self employed:")
print (df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed',data=df, palette= 'Set1')


# In[24]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


x=df.iloc[:,np.r_[1:5,9:11,13:15]].values
y=df.iloc[:,12].values


# In[28]:


x


# In[29]:


y


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[51]:


df


# In[ ]:





# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(x,y,  test_size=0.2, random_state=0)


from sklearn.preprocessing import LabelEncoder
Labelencoder_x= LabelEncoder()


# In[33]:


for i in range(0,5):
    X_train[:,i]=Labelencoder_x.fit_transform(X_train[:,i])
    X_train[:,7]=Labelencoder_x.fit_transform(X_train[:,7])
    
X_train


# In[38]:


Labelencoder_y=LabelEncoder()
y_train= Labelencoder_y.fit_transform(y_train)
y_train


# In[50]:


for i in range(0,5):
    X_test[:,i]=Labelencoder_x.fit_transform(X_test[:,i])
    X_test[:,7]=Labelencoder_x.fit_transform(X_test[:,7])
X_test


# In[64]:


Labelencoder_y=LabelEncoder()
y_test= Labelencoder_y.fit_transform(y_test)




y_test


# In[56]:


from sklearn.preprocessing import StandardScaler

ss =StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)


# In[58]:


from sklearn.ensemble import RandomForestClassifier

rf_clf= RandomForestClassifier()
rf_clf.fit(X_train,y_train)


# In[65]:


from sklearn import metrics
y_pred= rf_clf.predict(X_test)

print("acc of random forest clf is ",metrics.accuracy_score(y_pred,y_test))

y_pred


# In[ ]:





# In[ ]:





# In[77]:


from sklearn.naive_bayes import GaussianNB

nb_clf= GaussianNB()
nb_clf.fit(X_train,y_train)





y_pred= rf_clf.predict(X_test)

print("acc of GaussianNB is ",metrics.accuracy_score(y_pred,y_test))

y_pred


# In[ ]:





# In[80]:


from sklearn.tree import DecisionTreeClassifier
dt_clf=DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)


# In[82]:


y_pred=dt_clf.predict(X_test)
print('acc of DT is' ,metrics.accuracy_score(y_pred, y_test))


# In[ ]:




