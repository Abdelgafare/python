#!/usr/bin/env python
# coding: utf-8

# In[1]:


##libraries 
import pandas as pd
import numpy as np
df=pd.read_excel('C:\\Users\\pc\\Desktop\\Customer Call List.xlsx')
df.head(25)


# In[2]:


## data view 
df.info()


### we see that there some missings and some errors in phone number and in the last name  and our variables contian one boleean 
### yes or no


# In[3]:


### so lets see the missing values
print(df.isna().sum())
### we have 1 missing in last name and 2 in phone number and 4 in dont contact and different formats


# In[4]:


### checking duplicates and droping them 
df_no_duplicates = df.drop_duplicates()
df_no_duplicates.head(25)


# In[5]:


### first we will drop the Not_Useful_Column 

new_df=df_no_duplicates.drop(['Not_Useful_Column'],axis=1)
new_df


# In[6]:


## now we need to standardize this last two columns
new_df['Paying Customer']=new_df['Paying Customer'].replace({'Y': 'Yes', 'N': 'No'})
new_df
new_df['Do_Not_Contact']=new_df['Do_Not_Contact'].replace({'Y': 'Yes', 'N': 'No'})
new_df
### they are standardized but we still have the same problem as  missing values 


# In[20]:


### we will remove nan from the last columns 
### since we can use this to remove all the rows the contain missing values 


new_df = new_df.replace('N/a','')
new_df = new_df.replace('NaN','')
new_df=new_df.fillna('')
new_df


# In[7]:


### now we will  try to clean the  last name
new_df['Last_Name']=new_df['Last_Name'].str.strip('/\12.-_')
new_df


# In[8]:


### now we will do replace method for the phone numbers trying to standarize it in he

new_df["Phone_Number"] = new_df["Phone_Number"].str.replace('[^a-zA-Z0-9]','', regex=True)
new_df


# In[9]:


## making same format for phone number to be easy to read 
new_df["Phone_Number"] = new_df["Phone_Number"].apply(lambda x: str(x))

new_df["Phone_Number"] = new_df["Phone_Number"].apply(lambda x: x[0:3] + '-' + x[3:6] + '-' + x[6:10])

new_df
new_df["Phone_Number"] = new_df["Phone_Number"].str.replace('nan--','')

new_df["Phone_Number"] = new_df["Phone_Number"].str.replace('Na--','')
new_df


# In[10]:


### expanding address into three columns  and removing the adress one cause theres no need for it 
new_df[["Street_Address", "State", "Zip_Code"]] = new_df["Address"].str.split(',', n=2,  expand=True)

new_df
new_df=new_df.drop(['Address'],axis=1)
new_df


# In[11]:


### last check for missing 
### we will remove nan from the last columns 
### since we can use this to remove all the rows the contain missing values 


new_df = new_df.replace('N/a','')
new_df = new_df.replace('NaN','')
new_df=new_df.fillna('')
new_df



# In[12]:


print(new_df.isna().sum())


# In[13]:


### now we will drop the row that contain the empty  phone number 
for x in new_df.index:
    if new_df.loc[x, "Phone_Number"] == '':
        new_df.drop(x, inplace=True)

new_df
### or
#new_df = new_df.dropna(subset="Phone_Number"), inplace=True)


# In[14]:


### rearrange the data by the index 
new_df = new_df.reset_index(drop=True)
new_df


# In[16]:


new_df.to_csv('customer  cs after.csv', index=False)


# In[ ]:




