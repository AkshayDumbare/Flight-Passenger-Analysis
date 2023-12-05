#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data1=pd.read_csv("D:\\Data science\\Project\\Flight Passenger analysis using Python\\passenger data.csv")


# In[3]:


data1.head()


# In[4]:


data1.info()


# In[5]:


data1.isnull().sum()


# In[6]:


data2=pd.read_csv("D:\\Data science\\Project\\Flight Passenger analysis using Python\\survery data.csv")


# In[7]:


data2.head()


# In[8]:


data2.info()


# In[9]:


data2.isnull().sum()


# In[10]:


df=pd.merge(data1,data2)


# In[11]:


df.head()


# In[12]:


df.isnull().sum()


# In[13]:


df=df.dropna()


# In[14]:


df.shape


# In[15]:


df.drop_duplicates(inplace=True)


# In[16]:


df.shape


# In[17]:


df=df.set_index('id')


# In[18]:


df['satisfaction'] = df['satisfaction'].map(lambda x : 'Yes' if (x== 'satisfied') else 'No')


# In[19]:


df.columns=df.columns.str.title()


# In[20]:


df.head()


# In[ ]:





# In[21]:


df.shape


# In[22]:


df.tail()


# In[23]:


df.info()


# In[24]:


df.isnull().sum()


# In[25]:


df.describe()


# In[ ]:





# In[26]:


# Univariate Analysis


# In[27]:


df['Gender'].value_counts()


# In[28]:


a2=df['Satisfaction'].value_counts()
a2=a2[::-1]


# In[29]:


plt.figure(figsize=(8,6))
sns.countplot(x=df['Satisfaction'])
plt.title('Satisfaction Count')
for i, v in enumerate(a2):
    plt.text(i, v, str(v), ha='center')
plt.show()


# In[30]:


a1=df['Type Of Travel'].value_counts().rename_axis('Travel_Type').reset_index(name='count')
a1


# In[79]:


your_explode=[0, 0]
plt.pie(a1['count'], labels=a1.Travel_Type, autopct='%1.5f%%', explode=(0,0.05), startangle=90)
plt.show()


# In[32]:


a3=df['Customer Type'].value_counts()
a3


# In[33]:


sns.countplot(df['Customer Type'])
for i, v in enumerate(a3):
    plt.text(i, v, str(v), ha='center')
plt.show()


# In[34]:


df['Inflight Wifi Service'].value_counts()


# In[35]:


df['Departure/Arrival Time Convenient'].value_counts()


# In[36]:


df['Ease Of Online Booking'].value_counts()


# In[37]:


df['Inflight Entertainment'].value_counts()


# In[38]:


a4=df['Class'].value_counts()
a4


# In[39]:


sns.countplot(x=df['Class'])
for i, v in enumerate(a4):
    plt.text(i, v, str(v), ha='center')
plt.show()


# In[40]:


# Here we can see that most of the passenger travel through business and eco class


# In[41]:


df['Cleanliness'].value_counts()


# In[42]:


df['Seat Comfort'].value_counts()


# In[43]:


df['Inflight Service'].value_counts()


# In[44]:


df['Food And Drink'].value_counts()


# In[45]:


df['Checkin Service'].value_counts()


# In[46]:


df['On-Board Service'].value_counts()


# In[ ]:





# In[47]:


# Bivariate analysis


# In[ ]:





# In[48]:


x=pd.crosstab(df['Gender'], df['Satisfaction'])
x


# In[98]:


fig=plt.figure(figsize=(10,4))
a6=sns.countplot(x=df['Gender'],hue=df["Satisfaction"],palette='tab10')

for i in a6.containers:
    a6.bar_label(i)
plt.title("Gender Satisfaction Distribution")


# In[100]:


fig=plt.figure(figsize=(10,4))
a7=sns.countplot( x=df["Class"], hue=df["Satisfaction"], palette='viridis')
for bars in a7.containers:
    a7.bar_label(bars)
plt.title("Class wise Satisfaction")


# In[52]:


sns.countplot( x=df["Customer Type"], hue=df["Class"], palette='cubehelix')


# In[105]:


sns.displot(data=df, x="Flight Distance", hue="Satisfaction", kind="kde", multiple="stack")


# In[ ]:





# In[54]:


sns.countplot( x=df["Customer Type"], hue=df["Satisfaction"], palette='rocket')


# In[55]:


sns.displot(data=df, x="Class", hue="Inflight Wifi Service", multiple="dodge", palette='flare')


# In[56]:


sns.countplot(x=df['Customer Type'],hue=df["Cleanliness"])


# In[57]:


sns.displot(data=df,x='Customer Type',hue="Ease Of Online Booking", col="Gender", multiple="dodge")


# In[58]:


sns.displot(data=df,x='Customer Type',hue="Cleanliness", col="Type Of Travel", multiple="dodge")


# In[59]:


sns.barplot(data=df, x='Inflight Service', y='Flight Distance')


# In[60]:


sns.displot(data=df, x="Flight Distance", hue="Type Of Travel", kind="kde",col="Customer Type", multiple="stack")


# In[ ]:





# In[61]:


avg_Age=df.groupby(['Class', 'Gender'], as_index=False)['Age'].mean().sort_values(by='Age', ascending = False)
sns.set(rc={'figure.figsize':(15,6)})
sns.barplot(data=avg_Age, x='Class', y='Age', hue='Gender', palette='cubehelix')


# # conclusion
#  We can see that most of the people who are travel through the business class they are satisfied.
#  But the people who are travel with Eco class are not satisfied.
#  Also We can see most of the passenger prefferd short distance flight.

# In[ ]:





# In[62]:


import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=['Gender','Customer Type','Type Of Travel','Satisfaction','Class'])

df1 = encoder.fit_transform(df)


# In[63]:


df1.head()  
    


# In[64]:


df1.corr()


# In[65]:


df.info().describe()

