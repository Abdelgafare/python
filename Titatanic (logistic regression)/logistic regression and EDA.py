#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro


# In[3]:


### exploring the data
df=pd.read_csv('C:\\Users\pc\Desktop\Titatanic (logistic regression)\\titanic_train.csv')
df.head(10)


# In[4]:


df.shape


# In[5]:


df.info()
#### look that we have some missing values here 


# In[6]:


print(df.isna().sum())
### we 177 in age and 687 in cabin 


# In[7]:


### visualiazing the null values 
sns.heatmap(df.isna())

## as we say before the null values are more in cabin than the age so since we have 177 in age and 687 in cabin
## we can remove the null values if and only if the the percentage of the null values are more than 30 % of the data so lets see


# In[8]:


### counting the percentages of the null values for age column 
((df['Age'].isna().sum())/891)*100
### 19% so we will not remove this column 


# In[9]:


### counting the percentages of the null values for cabin column 

(df['Cabin'].isna().sum()/len(df['Cabin']))*100
###77% we will remove this column


# In[10]:


###droping capin column
df.drop('Cabin',axis=1,inplace=True)
df.head(10)


# In[11]:


### find distribution for age column lets see first the graph for this variable
sns.displot(x="Age",data=df)
## looks normal distribution


# In[12]:


# using shapiro test for testing the normality
x=df["Age"]

stat, p = shapiro(x)
print('Shapiro-Wilk Test:')
print('Test Statistic:', stat)
print('P-value:', p)

alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
### If the p-value is greater than the predefined significance level, you cannot reject the null hypothesis.the daa follow
###normal


# In[13]:


##filling  age column we can fill it with the mean or mode or the median since the normality
## mode_age = df['Age'].mode()[0]

## median_age = df['Age'].median()


mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)


# In[14]:


### see if there any null values 
df['Age'].isna().sum()
### data cleaning process are done  


# In[15]:


#countplot of subrvived vs not  survived since that survival : Survival 0 = No, 1 = Yes

sns.countplot(x='Survived',data=df)


# In[16]:


### now lets see the difference between the survived males females
sns.countplot(x='Survived',data=df,hue='Sex')
### we see that the most survival are females


# In[17]:


## logistic model lets  see again the variables
df.dtypes


# In[18]:


## convert sex to numerical variable(dummy variable) male will take 1 and female will take 0
sex_dummies = pd.get_dummies(df['Sex'], prefix='Sex', drop_first=True)
sex_dummies = sex_dummies.astype(int)  # Convert dummy variables to integers (0 and 1)
df = pd.concat([df, sex_dummies], axis=1)
df = df.drop(columns=['Sex'])  # Optional: drop the original 'Sex' column

df.head(10)


# In[19]:


## droping the columns  Ticket and Embarked and Name
df.drop(['Name','Ticket','Embarked'],axis=1,inplace=True)
df.head(10)


# In[20]:


## determine x independent and y dependent variables 
x=df[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Sex_male']]
y=df['Survived']


# In[21]:


## data modelling 
#import train test split method
from sklearn.model_selection import train_test_split


# In[22]:


#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[23]:


#import Logistic  Regression
from sklearn.linear_model import LogisticRegression
#Fit  Logistic Regression 
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[24]:


#predict
predict=lr.predict(x_test)


# In[25]:


# confusion matrix 
from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])


# In[26]:


#import classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))


# In[27]:


###Precision is fine considering Model Selected and Available Data. Accuracy can be increased
###by further using more features  or by using other model

###Note:
###Precision : Precision is the ratio of correctly predicted positive observations to the total predicted positive observations
###Recall : Recall is the ratio of correctly predicted positive observations to the all observations in actual class 
###F1 score - F1 Score is the weighted average of Precision and Recall.


# In[ ]:





# In[ ]:




