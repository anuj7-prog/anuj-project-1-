#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
# 2. Read the files into numpy matrices
male = np.genfromtxt('nhanes_adult_male_bmx_2020.csv', delimiter=',',
skip_header=1)
female = np.genfromtxt('nhanes_adult_female_bmx_2020.csv',
delimiter=',', skip_header=1)
print("Data loaded successfully.")
print("Male dataset shape:", male.shape)
print("Female dataset shape:", female.shape)


# In[2]:


# Extract weights (1st column)
f_weights = female[:, 0]
m_weights = male[:, 0]
plt.figure(figsize=(10, 8))
# Female histogram (top)
plt.subplot(2, 1, 1)
plt.hist(f_weights, bins=30, color='skyblue', edgecolor='black')
plt.title('Female Weights Distribution')
plt.ylabel('Frequency')
plt.xlim(30, 180) # Setting identical limits
# Male histogram (bottom)
plt.subplot(2, 1, 2)
plt.hist(m_weights, bins=30, color='lightgreen', edgecolor='black')
plt.title('Male Weights Distribution')
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')
plt.xlim(30, 180)
plt.tight_layout()
plt.show()


# In[3]:


plt.figure(figsize=(8, 6))
plt.boxplot([f_weights, m_weights], labels=['Females', 'Males'])
plt.title('Weight Comparison: Females vs Males')
plt.ylabel('Weight (kg)')
plt.show()


# In[4]:


def get_stats(data, name):
    print(f"--- {name} Weight Stats ---")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Median: {np.median(data):.2f}")
    print(f"Std Dev: {np.std(data):.2f}")
    print(f"Skewness: {stats.skew(data):.2f}")
get_stats(f_weights, "Female")
get_stats(m_weights, "Male")


# In[5]:


# 6. Calculate BMI and add to female matrix
# Height is at index 1 (cm). We convert to meters.
f_height_m = female[:, 1] / 100
bmi = female[:, 0] / (f_height_m ** 2)
female = np.column_stack((female, bmi))
# 7. Standardize the female dataset (Z-scores)
col_means = np.mean(female, axis=0)
col_stds = np.std(female, axis=0)
zfemale = (female - col_means) / col_stds
print("Standardized zfemale dataset created. Shape:", zfemale.shape)


# In[6]:


# Select specific columns from zfemale
# 1: Height, 0: Weight, 6: Waist, 5: Hip, 7: BMI
cols_idx = [1, 0, 6, 5, 7]
col_names = ['Height', 'Weight', 'Waist', 'Hip', 'BMI']
# Convert just these columns to a pandas dataframe for easy plotting
df_zfemale = pd.DataFrame(zfemale[:, cols_idx], columns=col_names)
sns.pairplot(df_zfemale, corner=True, diag_kind='kde')
plt.show()
print("--- Pearson Correlation ---")
print(df_zfemale.corr(method='pearson').round(3))
print("--- Spearman Correlation ---")
print(df_zfemale.corr(method='spearman').round(3))


# In[7]:


# 9. Compute and add ratios
# Female ratios (waist=6, height=1, hip=5)
f_wht = female[:, 6] / female[:, 1]
f_whp = female[:, 6] / female[:, 5]
female = np.column_stack((female, f_wht, f_whp))
# Male ratios (waist=6, height=1, hip=5)
m_wht = male[:, 6] / male[:, 1]
m_whp = male[:, 6] / male[:, 5]
male = np.column_stack((male, m_wht, m_whp))


# In[8]:


# 10. Draw boxplot for the ratios
plt.figure(figsize=(10, 6))
data = [f_wht, m_wht, f_whp, m_whp]
labels = ['F Waist/Height', 'M Waist/Height', 'F Waist/Hip', 'M Waist/Hip']
plt.boxplot(data, labels=labels)
plt.title('Body Ratio Comparisons: Females vs Males')
plt.show()


# In[9]:


# bmi is at index 7 in zfemale
bmi_sort_idx = np.argsort(zfemale[:, 7])
lowest_5_idx = bmi_sort_idx[:5]
highest_5_idx = bmi_sort_idx[-5:]
print("--- Standardized Data for 5 Lowest BMI Profiles ---")
# setting print options so it's readable
np.set_printoptions(precision=3, suppress=True)
print(zfemale[lowest_5_idx])
print("--- Standardized Data for 5 Highest BMI Profiles ---")
print(zfemale[highest_5_idx])


# In[ ]:




