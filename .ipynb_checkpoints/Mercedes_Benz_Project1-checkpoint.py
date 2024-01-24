#!/usr/bin/env python
# coding: utf-8

# In[1]:


####  Reduce the time a Mercedes-Benz spends on the test bench.

#Problem Statement Scenario:
#Since the first automobile, the Benz Patent Motor Car in 1886, Mercedes-Benz has stood for important automotive innovations. These include the passenger safety cell with a crumple zone, the airbag, and intelligent assistance systems. Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium carmakers. Mercedes-Benz is the leader in the premium car industry. With a huge selection of features and options, customers can choose the customized Mercedes-Benz of their dreams.

To ensure the safety and reliability of every unique car configuration before they hit the road, the company’s engineers have developed a robust testing system. As one of the world’s biggest manufacturers of premium cars, safety and efficiency are paramount on Mercedes-Benz’s production lines. However, optimizing the speed of their testing system for many possible feature combinations is complex and time-consuming without a powerful algorithmic approach.

You are required to reduce the time that cars spend on the test bench. Others will work with a dataset representing different permutations of features in a Mercedes-Benz car to predict the time it takes to pass testing. Optimal algorithms will contribute to faster testing, resulting in lower carbon dioxide emissions without reducing Mercedes-Benz’s standards.

Following actions should be performed:

If for any column(s), the variance is equal to zero, then you need to remove those variable(s).
Check for null and unique values for test and train sets.
Apply label encoder.
Perform dimensionality reduction.
Predict your test_df values using XGBoost.


# In[147]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[148]:


df_train = pd.read_csv("C:/Users/swati/OneDrive/Desktop/ML_CourseEnd_Projects/Mercedes_benz/train.csv")


# In[149]:


df_test = pd.read_csv("C:/Users/swati/OneDrive/Desktop/ML_CourseEnd_Projects/Mercedes_benz/test.csv")


# In[150]:


df_train.head()


# In[151]:


df_test.head()


# In[152]:


df_train.shape


# In[153]:


df_test.shape


# In[154]:


df_train.info()


# In[155]:


print("Integer Types : ")
print(df_train.select_dtypes('int64').columns)


# In[156]:


print("Float Types : ")
print(df_train.select_dtypes('float').columns)


# In[157]:


print("Object Types : ")
print(df_train.select_dtypes(np.object).columns)


# In[158]:


#######  Checking null values for training set


# In[159]:


df_train.select_dtypes('int64').isna().sum()


# In[160]:


df_train.select_dtypes('float64').isna().sum()


# In[161]:


df_train.select_dtypes(np.object).isna().sum()


# In[162]:


####  Checking variance for integer columns


# In[166]:


features_int = df_train.iloc[:, 10:386]


# In[167]:


features_int


# In[168]:


from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.0)
selector = selector.fit_transform(features_int)


# In[169]:


selector


# In[170]:


selector.shape


# In[171]:


### Checking variance for float 


# In[172]:


features_float = df_train.iloc[:, 1].values.reshape(-1,1)


# In[173]:


from sklearn.feature_selection import VarianceThreshold
selector_float = VarianceThreshold(threshold=0.0)
selector_float= selector_float.fit_transform(features_float)


# In[174]:


selector_float.shape


# In[175]:


#### Check for unique values


# In[176]:


col_order = np.sort(df_train.X0.unique()).tolist()
plt.figure(figsize=(20,6))
sns.boxplot(x = 'X0', y = 'y', data = df_train, order=col_order)


# In[177]:



col_order = np.sort(df_train.X1.unique()).tolist()
plt.figure(figsize=(20,6))
sns.boxplot(x = 'X1', y= 'y', data = df_train, order=col_order)


# In[178]:


col_order = np.sort(df_train.X2.unique()).tolist()
plt.figure(figsize=(20,6))
sns.boxplot(x = 'X2', y = 'y', data = df_train, order = col_order)


# In[179]:


col_order = np.sort(df_train.X3.unique().tolist())
plt.figure(figsize=(20,6))
sns.boxplot(x = 'X3',y='y',data = df_train, order=col_order)


# In[180]:


####  If for any column(s), the variance is equal to zero, then you need to remove those variable(s).


# In[181]:


col_order = np.sort(df_train.X4.unique().tolist())
plt.figure(figsize=(20,6))
sns.boxplot(x = 'X4', y = 'y', data = df_train, order = col_order)


# In[182]:


col_order = np.sort(df_train.X5.unique().tolist())
plt.figure(figsize=(20,6))
sns.boxplot(x = 'X5',y = 'y', data = df_train, order = col_order)


# In[183]:


col_order = np.sort(df_train.X6.unique().tolist())
plt.figure(figsize=(20,6))
sns.boxplot(x = 'X6', y = 'y', data = df_train, order = col_order)


# In[184]:


#####  X4 has low variance so need to drop X4
df_train.drop('X4', axis = 1, inplace = True)


# In[185]:


### Checking null values for test set


# In[186]:


df_test.info()


# In[187]:


print("Integer Types: ")
print(df_test.select_dtypes(np.int64).columns)


# In[188]:


print("Object Types : ")
print(df_test.select_dtypes(np.object).columns)


# In[189]:


df_test.select_dtypes('int64').isna().sum()


# In[190]:


df_test.select_dtypes(np.object).isna().sum()


# In[191]:


#####  checking variance for integer columns


# In[192]:


df_test


# In[196]:


features_int_test = df_test.iloc[:, 9:386]


# In[197]:


features_int_test


# In[198]:


from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.0)
selector = selector.fit_transform(features_int)


# In[199]:


selector.shape


# In[200]:


#### Checking value for unique values for test dataset


# In[201]:


df_test.X0.sort_values().unique()


# In[202]:


df_test.X1.sort_values().unique()


# In[203]:


df_test.X2.sort_values().unique()


# In[204]:


df_test.X3.sort_values().unique()


# In[205]:


df_test.X4.sort_values().unique()


# In[206]:


df_test.X5.sort_values().unique()


# In[207]:


df_test.X6.sort_values().unique()


# In[208]:


df_test.X8.sort_values().unique()


# In[209]:


####  Apply label encoder for training set


# In[210]:


df_train


# In[211]:


df_train_label = df_train.iloc[:,1]


# In[212]:


df_train_label


# In[213]:


df_cat = df_train.iloc[:,2:9]


# In[214]:


df_cat


# In[215]:


from sklearn.preprocessing import LabelEncoder

for f in [ "X0", "X1", "X2", "X3", "X5", "X6", "X8"]:
    le = LabelEncoder()
    df_cat[f] = le.fit_transform(df_cat[f])


# In[216]:


df_cat


# In[218]:


df_rest = df_train.iloc[:,9:385]


# In[219]:


df_rest


# In[220]:


updated_train_set = df_cat.join(df_rest)


# In[221]:


updated_train_set


# In[222]:


###  Apply Label encoder for test data set


# In[223]:


df_test


# In[224]:


df_cat_test = df_test.iloc[:, 1:9]


# In[225]:


from sklearn.preprocessing import LabelEncoder

for f in [ "X0", "X1", "X2", "X3", "X5", "X6", "X8"]:
    le = LabelEncoder()
    df_cat_test[f] = le.fit_transform(df_cat_test[f])


# In[226]:


df_cat_test


# In[227]:


df_rest_test = df_test.iloc[:,9:385]


# In[228]:


df_rest_test


# In[230]:


updated_test_set = df_cat_test.join(df_rest_test)


# In[231]:


updated_test_set


# In[232]:


updated_test_set.drop('X4', axis= 1, inplace =True)


# In[238]:


updated_test_set


# In[239]:


####  Perform dimensionality reduction on training set

from sklearn.decomposition import PCA


# In[240]:


n_comp = 12
pca = PCA(n_components=n_comp, random_state=420)

pca2_results_train = pca.fit_transform(updated_train_set)


# In[241]:


pca2_results_train


# In[242]:


pca2_results_train.shape


# In[243]:


####  Perform dimensionality reduction on test data set


# In[244]:


pca2_results_train = pca.fit_transform(updated_train_set)


# In[245]:


pca2_results_test = pca.transform(updated_test_set)


# In[246]:


pca2_results_test


# In[247]:


#######  XGBoost 


# In[292]:


import xgboost as xgb
from sklearn import model_selection
seed = 50


# In[293]:


kfold = model_selection.KFold(n_splits=  15, shuffle = True, random_state=seed)
model = xgb.XGBRegressor()
##model.fit(pca2_results_train, df_train_label)
results = model_selection.cross_val_score(model, pca2_results_train, df_train_label, cv=kfold)


# In[294]:


results


# In[291]:


print(results.mean())


# In[ ]:





# In[295]:


######    Hyperparameter Tuning


# In[ ]:





# In[306]:


dtrain = xgb.DMatrix(pca2_results_train, label=df_train_label)
dtest = xgb.DMatrix(pca2_results_test)


# In[307]:


params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:linear',
}


# In[309]:


cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=400,
    seed=42,
    nfold=5,
    metrics={'rmse'},
    early_stopping_rounds=10
)
cv_results


# In[ ]:





# In[ ]:




