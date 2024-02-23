#!/usr/bin/env python
# coding: utf-8

# # EDA

# In[113]:


##packages 
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[114]:


df=pd.read_csv('C:\\Users\\pc\\Desktop\\housing prices\\housing.csv')
df.head(10)


# In[ ]:


df.info()
df.shape


# In[ ]:


df.describe()


# In[ ]:


## lets see ocean_proximity
df['ocean_proximity'].value_counts()


# In[115]:


## conveting it to dummy 
ocean_proximity_dummies = pd.get_dummies(df['ocean_proximity'], prefix='ocean_proximity', drop_first=True)
ocean_proximity_dummies = ocean_proximity_dummies.astype(int)  # Convert dummy variables to integers (0 and 1)
df = pd.concat([df, ocean_proximity_dummies], axis=1)
df = df.drop(columns=['ocean_proximity'])  

df.head(10)


# In[ ]:


### checking for missings 
df.isna().sum()


# In[ ]:


### counting the percentages of the null values for total_bedrooms column 

(df['total_bedrooms'].isna().sum()/len(df['total_bedrooms']))*100
### less than 30 


# In[ ]:


### find distribution for age column lets see first the graph for this variable
sns.displot(x="total_bedrooms",data=df)
## doesnt seem like a  normal distribution but have some outliers we can check the normality after removing the outliers
## btw it have some skewnees to the left too 


# In[ ]:


### remove the outliers 
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define the outlier threshold
threshold = 1.5

# Filter out outliers
df_cleaned = df[~((df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))).any(axis=1)]

# Display the cleaned DataFrame
print(df_cleaned)


# In[ ]:


###we can fill the na with the mean  or remove 
df_cleaned['total_bedrooms'].mean()


# In[ ]:


df_cleaned['total_bedrooms'].mode()


# In[116]:


sns.displot(x="total_bedrooms",data=df_cleaned)
### it still skewed to the left also the mean and the median and the mode not equal 


# In[117]:


from scipy.stats import shapiro
shapiro(df_cleaned['total_bedrooms'])
## shapiro say that the graph look normal in the two cases after and before removing the outliers so we declare the test as 


# In[118]:


### so we just will the na 
df = df.dropna()
df.isna().sum()


# In[119]:


df.head(10)


# In[120]:


## looking for the correlation 
df.corr(numeric_only = True)


# In[121]:


### lets visaulize the correlation
sns.heatmap(df.corr(numeric_only=True), annot = True)
plt.rcParams['figure.figsize'] = (40,5)

plt.show()


# In[122]:


corr_matrix=df.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# #  Modeling

# In[123]:


### identify our indep and dep variables in case of the full model 
X = df.drop(columns=['median_house_value'], axis=1).select_dtypes(include=np.number)
Y = df['median_house_value']
## splitting the data into train and test 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)
# loading the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# In[124]:


# Model summary
import statsmodels.api as sm
X_train = sm.add_constant(X_train)  # Add a constant for the intercept
model = sm.OLS(Y_train, X_train).fit()

# Get the summary
summary = model.summary()
print(summary)


# In[125]:


import statsmodels.api as sm

def backward_stepwise_selection(X, Y):
    selected_features = list(X.columns)
    while True:
        x_selected = X[selected_features]
        model = sm.OLS(Y, sm.add_constant(x_selected)).fit()
        pvalues = model.pvalues[1:]  # Exclude the constant term
        max_pvalue = pvalues.max()
        if max_pvalue > 0.05:  # Stopping criterion (adjust as needed)
            remove_feature = pvalues.idxmax()
            selected_features.remove(remove_feature)
        else:
            break
    return selected_features, model

selected_features, final_model = backward_stepwise_selection(X,Y)
print("Selected features:", selected_features)
print(final_model.summary())


# In[126]:


#predicting with OLS
Y_pred=regressor.predict(X_test)
performance = pd.DataFrame({'PREDICTIONS': Y_pred, 'ACTUAL VALUES':Y_test})
performance.head()
performance['error']=performance['ACTUAL VALUES']-performance['PREDICTIONS']
performance.head()


# In[127]:


#preparing data for plotting
performance.reset_index(drop=True, inplace=True) #inplace turns into a column
performance.reset_index(inplace=True)
performance.head()


# In[128]:


#plot the residuals
fig = plt.figure(figsize=(10,5))
plt.bar('index', 'error', data=performance[:50], color='black', width=0.3)
plt.ylabel('residuals')
plt.xlabel('observations')
plt.show()


# In[129]:


nicer_OLS=sm.OLS(Y_train,X_train).fit()
nicer_OLS.summary()

