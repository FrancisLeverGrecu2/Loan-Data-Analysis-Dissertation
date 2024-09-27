#!/usr/bin/env python
# coding: utf-8

# In[181]:


# Load Libraries 

import pandas as pd 
import numpy as np


# In[182]:


# Load data 

LoanData = pd.read_csv('Loan_default.csv')
LoanData.head()


# In[183]:


# Get general info --> list of attributes 

LoanData.info()


# In[184]:


# Dimensions 
LoanData.shape


# In[185]:


# Drop Unnecessary columns 

LoanData = LoanData.drop(['LoanID'], axis = 1)


# In[186]:


# Check null values --> count 

nan_count = LoanData.isnull().sum()

print('Number of NaN values: ', nan_count)


# In[187]:


# Interpret: There are no null values in the dataset


# In[188]:


# Formatting the data 

LoanData.head()


# In[189]:


#1 Drop commas in Income, LoanAmounnt
LoanData['Income'] = LoanData['Income'].str.replace(',', '',)
LoanData

LoanData['LoanAmount'] = LoanData['LoanAmount'].str.replace(',', '',)
LoanData

# Drop percentage from InterestRate 

LoanData['InterestRate'] = LoanData['InterestRate'].str.replace('%', '')
LoanData

# reformat Interest rate as percentage 
LoanData['InterestRate'] = LoanData['InterestRate'].astype(float)
LoanData['InterestRate'].dtype

# Convert Interest rate to decimal format
LoanData['InterestRate'] = LoanData['InterestRate'] / 100
LoanData

# convert Income to numerical format 
LoanData['Income'] = LoanData['Income'].astype(float)
LoanData['Income']

# convert LoanAmount to numerical format
LoanData['LoanAmount'] = LoanData['LoanAmount'].astype(float)


# # Exploratory Data Analysis (EDA)

# In[190]:


# Look at dependent variable classes 
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(LoanData['Default'])
plt.title('Count of Default Class in Unbalanced Dataset')


# ## Numerical Variable Analysis

# In[191]:


# Extract numerical variables 
num_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
                'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
LoanData_num = LoanData[num_features]
LoanData_num


# In[192]:


# General statistics 

LoanData_num.describe()


# In[193]:


# Interpret 

# Age : Mean = 43, Range: 18 to 69
# Income: Mean = 82.5K, Range: 15K to 150K
# LoanAmount: Mean = 127K, Range: 5K to 250K
# CreditScore: Mean = 574, Range: 300 to 850 --> range of Equifax Credit Score system
# MonthsEmployed: Mean = 60, Range: 0 to 120
# NumCreditLines: Mean = 2.5, Range: 1 to 4
# InterestRate: Mean = 13.5%, Range: 2% to 25%
# LoanTerm: Mean = 36 months, Range: 12 months to 60 months
#DTIRatio: Mean = 0.5, Range: 0.1 to 0.9


# In[194]:


# Outlier detection (done early to make sure structure does not change later )

# create function to calculate outliers 

# list of num variables already extracted in
# --> Num_features

def outlier_detection(df, columns):
    
    #iqr 
    iqr_values = {}
    
    # boundaries 
    boundaries = {}
    
    for col in columns: 
        
        q1 = np.percentile(df[col], 25)
        q3 = np.percentile(df[col], 75)
        iqr = q3 - q1
        iqr_values[col] = iqr
        
        lower_bound = q1 - 1.5*iqr_values[col]
        upper_bound = q3 + 1.5*iqr_values[col]
        boundaries[col] = [lower_bound, upper_bound]
        
    return boundaries


# In[195]:


outlier_detection(LoanData_num, num_features)


# In[196]:


# Reformat attributes from months to years 

# MonthsEmployed in Years
LoanData['YearsEmployed'] = LoanData['MonthsEmployed'] / 12
LoanData['YearsEmployed'] = round(LoanData['YearsEmployed'], 1)
LoanData['YearsEmployed'].head()

# LoanTerm in Years 
LoanData['LoanTerm_In_Years'] = LoanData['LoanTerm'] / 12
LoanData.head()


# In[197]:


# DTI Ratio --> Standard: Good if DTI < 36% (0.36)
## Create attribute that classifies loan applicant DTI ratio based on th standard 


LoanData['DTI_Classification'] = np.where((LoanData['DTIRatio'] <= 0.36), 'Good', 'Bad')
                                                             


# In[198]:


# Classify loan Credit Score 
CreditScoreMin = LoanData['CreditScore'].min()
CreditScoreMax = LoanData['CreditScore'].max()
print('Minimum Credit Score:', CreditScoreMin)
print('Maximum CreditScore:', CreditScoreMax)


# In[199]:


#Interpret: Credit Score ranges from 300 to 850 --> range of Equifax system 

# -> Making assumption that the credit score system is the one being used 

# so: Categorize Credit score based on equifax thresholds


# In[200]:


conditions = [
    (LoanData['CreditScore'] >=300) & (LoanData['CreditScore'] < 580),
    (LoanData['CreditScore'] >= 580) & (LoanData['CreditScore'] < 670),
    (LoanData['CreditScore'] >= 670) & (LoanData['CreditScore'] < 740),
    (LoanData['CreditScore'] >= 740) & (LoanData['CreditScore'] < 800),
    (LoanData['CreditScore'] >= 800)
    
    
]


# In[201]:


values = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']


# In[202]:


LoanData['CreditScoreRating'] = np.select(conditions, values)


# In[203]:


# Check 
LoanData['CreditScore'].groupby(LoanData['CreditScoreRating']).describe()


# In[204]:


# check data types 
LoanData.dtypes


# In[205]:


# Visual analysis of numerical attributes 

import matplotlib.pyplot as plt 

# Age distribution 

plt.hist(LoanData['Age'])
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Frequency of Age Values')


# In[206]:


# Interpret: Age distribution is uniform apart from at the tails 
# What if we categorize to Age Groups? 

#Age Range 

MinAge = LoanData['Age'].min()
MaxAge = LoanData['Age'].max()

print('Minimum Age:', MinAge)
print('Maximum Age:', MaxAge)


# In[207]:


conditions = [
    (LoanData['Age'] >=18) & (LoanData['Age'] < 30),
    (LoanData['Age'] >= 30) & (LoanData['Age'] < 40),
    (LoanData['Age'] >= 40) & (LoanData['Age'] < 50),
    (LoanData['Age'] >= 50) & (LoanData['Age'] < 60),
    (LoanData['Age'] >= 60)
    
    
]

values = ['18-30', '30-40', '40-50', '50-60', '60+']


LoanData['AgeGroup'] = np.select(conditions, values)


# In[208]:


# Verify

LoanData['Age'].groupby(LoanData['AgeGroup']).describe()


# In[209]:


# reproduce Age distribution but using groups 

# Define the correct order of categories
age_order = ['18-30', '30-40', '40-50', '50-60', '60+']

# Convert AgeGroup to a categorical type with an explicit order
LoanData['AgeGroup'] = pd.Categorical(LoanData['AgeGroup'], categories=age_order, ordered=True)

# Get value counts, ensuring they are ordered according to the specified categories
age_group_counts = LoanData['AgeGroup'].value_counts(sort=False)

# Plot the histogram as a bar chart
age_group_counts.plot(kind='bar')

# Add title and labels
plt.title('Age Group Classification Frequencies')
plt.xlabel('Age Group')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# In[210]:


# get values of Age distribution

LoanData['AgeGroup'].value_counts() # slightly more 18-30, could be young prof / students?


# In[211]:


# Income 

plt.hist(LoanData['Income']) # Income also uniformally distributed 


# In[212]:


# Group Income to make histogram more interpretable 

## Look at range of values to see how to split income 
IncomeMin = LoanData['Income'].min()
IncomeMax = LoanData['Income'].max()
print('Income Minimum:', IncomeMin)
print('Income Maximum:', IncomeMax)


# In[213]:


conditions = [
    (LoanData['Income'] >=15000) & (LoanData['Income'] < 30000),
    (LoanData['Income'] >= 30000) & (LoanData['Income'] < 60000),
    (LoanData['Income'] >= 60000) & (LoanData['Income'] < 90000),
    (LoanData['Income'] >= 90000) & (LoanData['Income'] < 150000)
    
    
    
]

values = ['Low', 'Lower-Middle', 'Middle', 'Upper']


LoanData['IncomeGroup'] = np.select(conditions, values)


# In[214]:


#Check
LoanData['Income'].groupby(LoanData['IncomeGroup']).describe()


# In[215]:


# Define the correct order of categories
Income_order = ['Low', 'Lower-Middle', 'Middle', 'Upper']

# Convert AgeGroup to a categorical type with an explicit order
LoanData['IncomeGroup'] = pd.Categorical(LoanData['IncomeGroup'], categories=Income_order, ordered=True)


# Get value counts, ensuring they are ordered according to the specified categories
Income_group_counts = LoanData['IncomeGroup'].value_counts(sort=False)

# Plot the histogram as a bar chart
Income_group_counts.plot(kind='bar')#Plot the Income Categories 

plt.hist(LoanData['IncomeGroup'])
plt.xlabel('Income Group')
plt.ylabel('Frequeny')
plt.title('Frequency of Income Groups')


# Note: Try and split by Default Class


# In[216]:


# Amount of defaults by Income Groups 

pd.crosstab(LoanData['IncomeGroup'], LoanData['Default'], normalize = 'columns')


# In[217]:


# Get Default Rates for the different income groups 

LowIncomeDefault = 6236 / (6236+22166)
LowMidIncomeDefault = 7249 / (7242+49409)
MidIncomeDefault = 5779 / (5779+51095)
UpperIncomeDefault = 10396 / (10396+103024)

print('Low Income Default Rate:', round(LowIncomeDefault, 3))
print('Low-Medium Income Default Rate', round(LowMidIncomeDefault, 3))
print('Mid Income Default Rate:', round(MidIncomeDefault, 3))
print('Upper Income Default Rate:', round(UpperIncomeDefault, 3))


# In[218]:


# Loan Amount analysis

plt.hist(LoanData['LoanAmount']) # similar to Income


# In[219]:


# Define the correct order of categories
Credit_order = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']

# Convert AgeGroup to a categorical type with an explicit order
LoanData['CreditScoreRating'] = pd.Categorical(LoanData['CreditScoreRating'], categories=Credit_order, ordered=True)


# Get value counts, ensuring they are ordered according to the specified categories
Credit_group_counts = LoanData['CreditScoreRating'].value_counts(sort=False)

# Plot the histogram as a bar chart
Income_group_counts.plot(kind='bar')#Plot the Income Categories 

plt.hist(LoanData['CreditScoreRating'])
plt.xlabel('Credit Group')
plt.ylabel('Frequeny')
plt.title('Frequency of Credit Groups')


# Note: Try and split by Default Class


# In[220]:


# analyse the income of the different credit score classes 

crosstab1 = pd.crosstab(LoanData['IncomeGroup'], LoanData['CreditScoreRating'], normalize = 'columns')
crosstab1


# In[221]:


# bar plot of crosstab1 

barplot1 =crosstab1.plot.bar(rot=0)
barplot1


# In[222]:


# Average Credit Score rating by Income Group

LoanData['CreditScore'].groupby(LoanData['IncomeGroup']).mean()


# In[223]:


# Median Credit Score rating by Income Group 

LoanData['CreditScore'].groupby(LoanData['IncomeGroup']).median()


# In[224]:


# Interpret: On average, the different income groups have the same avg and median credit score ratings 


# In[225]:


group = LoanData['Default']

sns.histplot(data = LoanData,x = 'IncomeGroup', hue = group, multiple = 'dodge')
plt.xlabel('Income Group')
plt.title('Default Across Income Groups')




# In[226]:


# Interpret: more or les same amount of defaults in each income group 


# In[227]:


# Amount of defaults across credit score groupinngs 

sns.histplot(x = LoanData['CreditScoreRating'], hue = group, multiple ='dodge')
plt.title('Credit Score Group Across Default Groups')


# In[228]:


# Interpret: There appears to be more defaults (absolute quantity) in low credits score group

pd.crosstab(LoanData['CreditScoreRating'], LoanData['Default'], normalize = 'columns')


# In[229]:


# Interpret: The Default and No Default group have similar distributions of credit score rating



# In[230]:


# Years Employed

# general distribution --> uniform

sns.histplot(LoanData, x= 'YearsEmployed', bins = 5, hue = 'Default', multiple = 'dodge') # years employed uniformally distributed


# In[231]:


# statistics of years employed by default group 

LoanData['YearsEmployed'].groupby(LoanData['Default']).describe()


# In[232]:


# Interpret: Range the same. Slight different in Mean 


# In[233]:


# Median years employed by default group

LoanData['YearsEmployed'].groupby(LoanData['Default']).median() # more significance between groups medians


# In[234]:


# Num Credit Lines 

sns.histplot(LoanData['NumCreditLines']) # evenly distributed 


# In[235]:


# is there a difference in the number of credit lines for the different default groups? 

LoanData['NumCreditLines'].groupby(LoanData['Default']).describe() 

# more or less the same, although median is higher for the default group


# In[236]:


# LoanTerm

sns.histplot(LoanData['LoanTerm_In_Years']) # Equally distributed across all term lengths

# Interpret: These are mainly short term loans 


# In[237]:


# Do loan terms differ across default groups? 

pd.crosstab(LoanData['LoanTerm_In_Years'], LoanData['Default'], normalize = 'columns') # same distribution


# In[238]:


# Do loan terms differ across income groups (do you get diff loan terms/conditions based on income group?)

pd.crosstab(LoanData['LoanTerm_In_Years'], LoanData['IncomeGroup'], normalize = 'columns')

# Interpret: same distribution across income groups 


# In[239]:


# Do Loan term differ across income groups? (do you get diff loan terms/conditions based on credit score?)

pd.crosstab(LoanData['LoanTerm_In_Years'], LoanData['CreditScoreRating'],  normalize = 'columns')

# Interpret: more or loss the same across all credit score groups


# In[240]:


# DTI Ratio

# --> Look at the DTI Ratio gorups 

sns.histplot(LoanData['DTI_Classification']) 
plt.title('Number of Applicants in each DTI Ratio Class')
plt.xlabel('DTI Class')

# there are significantly more bad DTI ratios than good 


# In[241]:


# Does it differ among Default groupings? 

sns.histplot(x = LoanData['DTI_Classification'], hue = group, multiple = 'dodge')
plt.title('DTI Count by Default Variable Class')
plt.xlabel('DTI Class')


# In[242]:


# what is the proportion across default groups? 

pd.crosstab(LoanData['DTI_Classification'], LoanData['Default'], normalize = 'columns')

# proportions the same across default groups 


# In[243]:


# what is the proportion across income groups? 

pd.crosstab(LoanData['DTI_Classification'], LoanData['IncomeGroup'], normalize = 'columns')

# proportions the same across income groups 


# In[244]:


# what is the proportion across credit score groups? 

pd.crosstab(LoanData['DTI_Classification'], LoanData['CreditScoreRating'], normalize = 'columns')

# same across credit score groups 


# In[245]:


# DTI Ratio across age groups 

pd.crosstab(LoanData['DTI_Classification'], LoanData['AgeGroup'], normalize = 'columns')

# same across age groups


# ## Categorical Variable Analysis

# In[246]:


# Education levels of loan applicants 

sns.histplot(data = LoanData, x = 'Education')


# In[247]:


# Education level filtered by default group 

sns.histplot(data = LoanData, x = 'Education', hue = group, multiple = 'dodge')
plt.title('Education Level across Default Group')

# interpret: Same amount of education across group


# In[248]:


# crosstab education / default

pd.crosstab(LoanData['Education'], LoanData['Default'], normalize = 'columns')

# Interpret: Minor differences between the two default groups 
# --> maybe as education increases default decreases? 


# In[249]:


# How does income group differ across education levels? 

pd.crosstab(LoanData['IncomeGroup'], LoanData['Education'], normalize = 'columns')

# Interpret: similar income distributions across education levels


# In[250]:


# How does education level differ across age groups?

pd.crosstab(LoanData['AgeGroup'], LoanData['Education'], normalize = 'columns')

# Education is split the same way across Age groups


# In[251]:


# Are the 'less educated' more or less likely to have dependents? 

pd.crosstab(LoanData['HasDependents'], LoanData['Education'], normalize = 'columns')

# nothing stands out --> minimal differences in probability of having kids 


# In[252]:


# What are the loan purposes of the different education levels? 

pd.crosstab(LoanData['LoanPurpose'], LoanData['Education'], normalize = 'columns')

# more or less the same across education groups


# In[253]:


# Employment Type 

# Are there differences in employment type across the income groups?
pd.crosstab(LoanData['EmploymentType'], LoanData['IncomeGroup'], normalize = 'columns')

# more or less the same


# In[254]:


# Are the differences in employment type across age groups? 

pd.crosstab(LoanData['AgeGroup'], LoanData['EmploymentType'], normalize = 'columns')


# In[255]:


# Does Employment type have an effect on DTI_classification? 

pd.crosstab(LoanData['EmploymentType'], LoanData['DTI_Classification'], normalize = 'columns')

# not substantial differences 


# In[256]:


# Does Employment type make a difference on whether individuals have mortgages? 

pd.crosstab(LoanData['EmploymentType'], LoanData['HasMortgage'], normalize = 'columns')

# no substantial differences


# In[257]:


# Marital Status 

## distribution 

sns.histplot(LoanData['MaritalStatus'])

# same proportion across marital status 


# In[258]:


# Do income levels differ dependent on marital status? 

pd.crosstab(LoanData['MaritalStatus'], LoanData['IncomeGroup'], normalize = 'columns')

# same frequencies for all income levels across relationship status


# In[259]:


# Do the married individuals have more chance of haivng dependents? 

pd.crosstab(LoanData['MaritalStatus'], LoanData['HasDependents'], normalize = 'columns')

# same probabilities across


# In[260]:


# Are married individuals more likely to default? 

sns.histplot(LoanData, x= 'MaritalStatus', hue = group, multiple = 'dodge')

# uniformally distributed across 'Default' classs


# In[261]:


# calculate the probabilities of default for different Relationship statuses 

pd.crosstab(LoanData['MaritalStatus'], LoanData['Default'], normalize = 'columns')


# In[262]:


# Look defaults among marital statuses

pd.crosstab(LoanData['MaritalStatus'], LoanData['Default'], normalize = 'columns')

# More divorces if default --> could be that divorces cause defaults as loans were taken jointly when married?


# In[263]:


# Look at age groups of different marital statuses 

pd.crosstab(LoanData['MaritalStatus'], LoanData['AgeGroup'], normalize = 'columns')

# distribution is the same for all age groups


# In[264]:


# Do DTI ratios differ among relationoships statuses? (Again, does divorce affect loans?)

pd.crosstab(LoanData['MaritalStatus'], LoanData['DTI_Classification'], normalize = 'columns')

# no difference among diff marital statuses 


# In[265]:


# Are there significant differences in CreditScore ratings among relat. statuses?

pd.crosstab(LoanData['MaritalStatus'], LoanData['CreditScoreRating'], normalize = 'columns')

# no real differences


# In[266]:


# HasMortgage 

# Count 

sns.histplot(LoanData['HasMortgage'])
# same frequency


# In[267]:


# What if we filter by Default? 

sns.histplot(LoanData, x = 'HasMortgage', hue = group, multiple = 'dodge')

#Same proportion across Default filter


# In[268]:


# What if we filter by Income grouping? 

pd.crosstab(LoanData['HasMortgage'], LoanData['IncomeGroup'], normalize = 'columns')

# same distribution for different income groups


# In[269]:


# Looking at HasMortgage across age groups: 

pd.crosstab(LoanData['HasMortgage'], LoanData['AgeGroup'], normalize = 'columns')

# same distribution across the different age groups 


# In[270]:


# Maybe there is a relationship between having a mortgage and credit score? 

pd.crosstab(LoanData['HasMortgage'], LoanData['CreditScoreRating'], normalize = 'columns')

# unlikely as there appears to be no substantial differences between the groups


# In[271]:


# HasDependents 

# General distribution 

sns.histplot(LoanData, x = 'HasDependents')

# Interpret: Uniform distributionabs


# In[272]:


# Is there a difference in proportion of 'has dependents' among Default class?

sns.histplot(LoanData, x = 'HasDependents', hue = group, multiple = 'dodge')

# same distribution across


# In[273]:


# Proba of having dependents among default status 

pd.crosstab(LoanData['HasDependents'], LoanData['Default'], normalize = 'columns')

# minor differences among default groups 


# In[274]:


# Income levels of the HasDependents groups 

pd.crosstab(LoanData['HasDependents'], LoanData['IncomeGroup'], normalize = 'columns')

# same probabilities across diff income groups 


# In[275]:


# What about across age groups? 

pd.crosstab(LoanData['HasDependents'], LoanData['AgeGroup'], normalize = 'columns')


# In[276]:


# Loan purpose

# distribution 

sns.histplot(LoanData['LoanPurpose']) # equal distribution


# In[277]:


# Are there differences in the loan purposes between those that default and those 
# that don't? 

sns.histplot(LoanData, x = 'LoanPurpose', hue = group, multiple = 'dodge')

# Interpret: same partitioning across Default and NoDefault


# In[278]:


# Do people of diff. income groups get loans for diff reasons? 

pd.crosstab(LoanData['LoanPurpose'], LoanData['IncomeGroup'], normalize = 'columns')

# no real differences


# In[279]:


# Do people from different age groups get loans for different reasons? 

pd.crosstab(LoanData['LoanPurpose'], LoanData['AgeGroup'], normalize = 'columns')

# no real differences 


# In[280]:


# Are people more likely to default because of one loan type than another?

pd.crosstab(LoanData['LoanPurpose'], LoanData['Default'], normalize = 'columns')

# maybe slight difference with homes but other than that no real variability


# In[281]:


# HasCoSigner

# General distribution

sns.histplot(LoanData['HasCoSigner']) # same count of both 


# In[282]:


# Is there a differences between the default groups? 

sns.histplot(LoanData, x= 'HasCoSigner', hue = group, multiple = 'dodge')

# interpret: For both groups the proba. seem to be the same 


# In[283]:


# check 

pd.crosstab(LoanData['HasCoSigner'], LoanData['Default'], normalize = 'columns')

# those that do not have Cosigner may be slightly more likely to default? 
# Theory: maybe those with Cosigner feel social pressure given loan will fall on 
# Cosigner if they fail to repay 


# # General conclusions from initial EDA

# In[284]:


#  Default imbalanced --> need to rebalance to be able to model properly 
# No substantial differences across the independent variables for the two classes 
# How is this possible? --> Synthetic dataset --> likely to be poorly designed
# No real dependencies in the data


# # Balance the dataset --> SMOTE

# In[285]:


# Conversion of single class attributes 

# applicable for : HasMortgage, HasDependents, HasCoSigner

# Make changes on copy of data in case 
LoanData2 = LoanData
LoanData2.head()



# In[286]:


LoanData2.HasMortgage.replace(('Yes', 'No'), (1, 0), inplace=True)


# In[287]:


LoanData2.HasDependents.replace(('Yes', 'No'), (1, 0), inplace=True)
LoanData2.HasCoSigner.replace(('Yes', 'No'), (1, 0), inplace=True)


# In[288]:


# convert remaining categorical variables using get_dummies

Dummies = pd.get_dummies(LoanData2[['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']])


# In[289]:


# Concat dummied columns
LoanData2 = pd.concat([LoanData2, Dummies], axis = 1)


# In[290]:


LoanData2.describe()


# In[291]:


# Extract numerical attributes for transformation

# check data types 
LoanData2.dtypes

LoanData2 = LoanData2._get_numeric_data()
LoanData2


# In[292]:


# SMOTE Transformation 

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 

X = LoanData2.drop(['Default'], axis = 1)
y = LoanData2['Default']

sm = SMOTE(random_state = 42)
X_res, y_res = sm.fit_resample(X, y)



# In[293]:


# Represent graphically 

sns.histplot(y_res)
plt.title('Count of Default Class on Balanced Dataset')


# In[294]:


# Attach X_res, y_res to assemble new dataset 

LoanData2 = pd.concat([X_res, y_res], axis = 1)
LoanData2.shape


# In[295]:


LoanData2['Default'].value_counts()


# # Numerical Analysis Part 2

# In[296]:


# re-perform outlier detection on new data 

outlier_detection(LoanData2, num_features)


# In[297]:


# Interpret: no real outliers to worry about 


# In[298]:


# Age 

# visualise distribution of age 

sns.histplot(LoanData2, x= 'Age', bins = 7)
plt.title('Count of Loan Applicants in each Age Group')


# In[299]:


# Interpret: Age is mildly skewed --> more young applicants than elder 
# --> more realistic 


# In[300]:


# How does age group differ for those that default vs those that do not default?

# reset group seperator 
group = LoanData2['Default']

sns.histplot(LoanData2, x = 'Age', bins = 7, hue = group, multiple = 'dodge')
plt.title('Count of Applicants in Each Age Group by Default Class')


# In[301]:


# Interpret: There are more defaults among younger loan applicants 

# --> seems more realistic --> younger loan applicants are probably 
# more likely to default as they have lower income (in reality, not necessarily applicable 
# to this dataset)


# In[302]:


# Get probability 

# reapply age group categorization 

conditions = [
    (LoanData2['Age'] >=18) & (LoanData2['Age'] < 30),
    (LoanData2['Age'] >= 30) & (LoanData2['Age'] < 40),
    (LoanData2['Age'] >= 40) & (LoanData2['Age'] < 50),
    (LoanData2['Age'] >= 50) & (LoanData2['Age'] < 60),
    (LoanData2['Age'] >= 60)
    
    
]

values = ['18-30', '30-40', '40-50', '50-60', '60+']


LoanData2['AgeGroup'] = np.select(conditions, values)

# reapply income group categorization
conditions = [
    (LoanData2['Income'] >=15000) & (LoanData2['Income'] < 30000),
    (LoanData2['Income'] >= 30000) & (LoanData2['Income'] < 60000),
    (LoanData2['Income'] >= 60000) & (LoanData2['Income'] < 90000),
    (LoanData2['Income'] >= 90000) & (LoanData2['Income'] < 150000)
    
    
    
]

values = ['Low', 'Lower-Middle', 'Middle', 'Upper']


LoanData2['IncomeGroup'] = np.select(conditions, values)



pd.crosstab(LoanData2['AgeGroup'], LoanData2['Default'], normalize = 'columns')


# In[303]:


# Interpret: Significantly more defaults among 18-30 AgeGroup
# As Age increases, probability of default decreases 


# In[304]:


# How is income group distributed across age group 

pd.crosstab(LoanData2['AgeGroup'], LoanData2['IncomeGroup'], normalize = 'columns')


# In[305]:


# Visualize Income distribution among age groups 

sns.histplot(LoanData2, x = 'AgeGroup', hue = LoanData2['IncomeGroup'], multiple = 'dodge')


# In[306]:


sns.histplot(LoanData2, x = 'IncomeGroup')
plt.title('Count of applicants in each income group')


# In[307]:


# get values 

pd.crosstab(LoanData['IncomeGroup'], LoanData['AgeGroup'], normalize = 'columns')


# In[308]:


# Interpret: No significant differences between age gorups-->proportions seem to be the same 


# In[309]:


# Look at CreditScoreRating by age group

# Recreate credit score rating system
conditions = [
    (LoanData2['CreditScore'] >=300) & (LoanData2['CreditScore'] < 580),
    (LoanData2['CreditScore'] >= 580) & (LoanData2['CreditScore'] < 670),
    (LoanData2['CreditScore'] >= 670) & (LoanData2['CreditScore'] < 740),
    (LoanData2['CreditScore'] >= 740) & (LoanData2['CreditScore'] < 800),
    (LoanData2['CreditScore'] >= 800)
    
    
]

values = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']

LoanData2['CreditScoreRating'] = np.select(conditions, values)


# In[310]:


# Credit score rating 

sns.histplot(LoanData2['CreditScoreRating'])


# In[311]:


#CreditScoreRating by Age Group 

pd.crosstab(LoanData['AgeGroup'], LoanData['CreditScoreRating'], normalize = 'columns')



# In[312]:


# Plot

groups2 = LoanData['AgeGroup']

sns.histplot(data = LoanData, x = 'CreditScoreRating', hue = 'AgeGroup', multiple = 'dodge')

# Interpret: Proportions are more or les the same 


# In[313]:


# Look at Age Group and HasDependents relationship

pd.crosstab(LoanData['AgeGroup'], LoanData['HasDependents'], normalize = 'columns')

# 18-30 seem to have slightly less kids but otherwise more or less the same 


# In[314]:


# Income

# General distribution 

sns.histplot(LoanData2['Income']) # more skewed to the left


# In[315]:


## Plot the income groups 

sns.histplot(LoanData2, x = 'IncomeGroup') # significant proportion of individuals have high income


# In[316]:


# Look at actuall Proportions 

(LoanData2['IncomeGroup'].value_counts()/451388) * 100


# In[317]:


# 40% are high income, 21% are middle, 23% are Lower-Middle, 15% are Low


# In[318]:


# Does this change depending on the default group of the individual? 

pd.crosstab(LoanData2['IncomeGroup'], LoanData2['Default'], normalize = 'columns')


# In[319]:


# Interpret: Income group distribution is different for the default groups 

# -->  Higher proportion of high income individuals in high income group (relative to no default group)


# In[320]:


# Represent graphically: 

sns.histplot(LoanData, x = 'IncomeGroup', hue = group, multiple = 'dodge')


# In[321]:


# Is average years employed different for the different income groups? 

LoanData['YearsEmployed'].groupby(LoanData['IncomeGroup']).mean() # same all across 


# In[322]:


# Median

LoanData['YearsEmployed'].groupby(LoanData['IncomeGroup']).median()


# In[323]:


# Add DTI_Classification to new dataset 

LoanData2['DTI_Classification'] = np.where((LoanData2['DTIRatio'] <= 0.36), 'Good', 'Bad')


# In[324]:


# Look at DTI Ratio rating differences for the different income groups 

pd.crosstab(LoanData2['IncomeGroup'], LoanData2['DTI_Classification'], normalize = 'columns')

# same results for the different DTI Ratio groups 


# In[325]:


# Do wealthier people have more credit lines? 

groups2 = LoanData2['IncomeGroup']

sns.histplot(LoanData2, x = 'NumCreditLines', hue = groups2, multiple = 'dodge' )

# proportions look the same 


# In[326]:


# check 

pd.crosstab(LoanData['IncomeGroup'], LoanData['NumCreditLines'], normalize = 'columns')

# same distributions across income groups


# In[327]:


# LoanAmount

# general distribution 

sns.histplot(LoanData2['LoanAmount'])


# In[328]:


# comppare to original data 

sns.histplot(LoanData['LoanAmount'])


# In[329]:


# Interpret: transformed data is slightly more skewed to the right 

# means in our modified data, more high income individuals apply for larger loans --> realistic


# In[330]:


# Look at it from IncomeGroup perspective 

sns.histplot(LoanData2, x = 'LoanAmount', hue = LoanData['IncomeGroup'], multiple = 'dodge')

# interpre: moore or less the same across


# In[331]:


# CreditScoreRating


# Distribution

sns.histplot(LoanData2['CreditScoreRating']) # same proportions as original data


# In[332]:


# Interpret: 
# Significant amount of people have poor credit score rating 


# In[333]:


# Does this differ for the different income groups? 

sns.histplot(LoanData2, x = 'CreditScoreRating', hue = LoanData2['IncomeGroup'], multiple = 'dodge')


# In[334]:


# interpet: 

# Proportions seem more or less the same at all credit score rating groups
# BUT 


# In[335]:


# Look at it numerically 

pd.crosstab(LoanData2['CreditScoreRating'], LoanData2['IncomeGroup'], normalize = 'columns')

# proportions very similar throughout


# In[336]:


# Are there differences in credit score rating between those that default and those that do not default 

sns.histplot(LoanData2 , x = 'CreditScoreRating', hue = group, multiple = 'dodge')

# Higher proportion of low credit score groups default than high credit score groups 


# In[337]:


# look numerically 

pd.crosstab(LoanData2['CreditScoreRating'], LoanData2['Default'], normalize = 'columns')


# In[338]:


# Interpret: As Credit Score increases, probability decreases ? 
# Proportion of people that default decreases as we go across the credit score rating classes 


# In[339]:


# How do credit score differ across age groups? 

sns.histplot(LoanData2, x = 'CreditScoreRating', hue = LoanData2['AgeGroup'], multiple = 'dodge')

# same proportions at each age group levels


# In[340]:


# YearsEmployed 

## distribution

sns.histplot(LoanData['YearsEmployed'])


# In[341]:


# How does it differ across age groups? 

sns.histplot(LoanData2, x = 'YearsEmployed', hue = LoanData2['AgeGroup'], multiple = 'dodge', bins = 5)


# same distribution for all years 


# In[342]:


# Could there be a relationship between years employed and income 

sns.histplot(LoanData2, x = 'YearsEmployed', hue = LoanData2['IncomeGroup'], multiple = 'dodge', bins = 5)

# proportions the same 


# In[343]:


# What do the average and median incomes look like for the years employed 

MedYE = LoanData2['Income'].groupby(round(LoanData2['YearsEmployed'])).median()
MeanYE = LoanData2['Income'].groupby(round(LoanData2['YearsEmployed'])).mean()

print('Median Income For Number of Years Employed:', MedYE)
print('Mean Income For Number of Years Employed:', MeanYE)


# In[344]:


# Interpret: 

#Median: 75k for most, but 79k at years ==0, 79k at years ==9, 82k at years ==10
#Mean: 77k for most, but 80k at years ==0, 80k at years ==9, 82k at years ==10

# --> these results do not make much sense 
# Expect the highest to be at or arround mployed 
# but income levels surprisingly high for the low emp. end 


# In[345]:


# NumCreditLines 

## general distribution 

sns.histplot(LoanData2, x = 'NumCreditLines') # slighly more people have 2 or 3 credit lines


# In[346]:


# are there income differences between those with more credit lines? 

sns.histplot(LoanData2, x = 'NumCreditLines', hue = LoanData2['IncomeGroup'], multiple = 'dodge', bins = 5)



# In[347]:


# Interpret: Same pattern throughout 


# In[348]:


# Is there a diff in the number of credit lines between default classes? 

sns.histplot(LoanData2 , x = 'NumCreditLines', hue = group, multiple = 'dodge', bins = 8)

# Out of those with 1 credit line --> split 50/50 across the two classes 
# Out of those with 2 credit lines --> 29% more applicants default 
# Out of those with 3 credit lines --> About 20% more people default
# Out of those with 4 credit lines --> significantly less people default 

# interpret: 1 credit lines --> about 50/50
# As nu. credit lines increase, the proba. of default increases 
# BUT this is only up to a point at which the people with the most 
# credit lines default significantly less 


# In[349]:


# Do people with higher credit score ratings have more credit lines? 

sns.histplot(LoanData2, x = round(LoanData2['LoanAmount']), 
             hue= LoanData2['CreditScoreRating'], multiple = 'dodge',
            bins = 5)

# Interpret: same pattern throughout


# In[350]:


# Do people that take out larger laons have more credit lines? 

sns.histplot(LoanData2, x = round(LoanData2['LoanAmount']),
            hue = LoanData2['NumCreditLines'], multiple = 'dodge',
            bins = 5)

# interpret: same pattern throughout


# In[351]:


# LoanTerm

## Distribution

sns.histplot(LoanData2, x = 'LoanTerm_In_Years', bins = 5)

# Loans are for 1 to 5 years A
# Slightly more loans are 2 to 4 years long 


# In[352]:


# Is there a difference in loan term length between default groups? 

sns.histplot(LoanData2, x = 'LoanTerm_In_Years', bins = 10, hue = group, multiple = 'dodge')

#Interpret: 

# Those that do not default are uniformally distributed across the LoanTerm dist.
# LoanTerm length iis a lot more varied among the default group


# In[353]:


# Maybe credit score rating affect LoanTerm length? 

sns.histplot(LoanData2, x = 'LoanTerm_In_Years', bins = 5, hue = LoanData2['CreditScoreRating'], multiple = 'dodge')

# same pattern throughout so unlikely to be a significant difference


# In[354]:


# Is there a difference in loan term duration between the different income groups? 

sns.histplot(LoanData2, x = 'LoanTerm_In_Years',
             hue = LoanData2['IncomeGroup'],
             multiple = 'dodge',
            bins = 5)


# In[355]:


# Are there any differences in the length of loan terms across the different Income groups? 

LoanData2['LoanTerm_In_Years'].groupby(LoanData2['IncomeGroup']).describe()

# same across all income groups


# In[356]:


# Are there any differences in the length of loan terms across the different AgeGroups? 

LoanData2['LoanTerm_In_Years'].groupby(LoanData2['AgeGroup']).describe()

# same across all age groups


# In[357]:


# DTI Ratio

## Distribution 

sns.histplot(LoanData2, x = 'DTIRatio', bins = 20) # seems fairly normally distributed


# In[358]:


# Is there a difference between the Default class types? 

sns.histplot(LoanData2, x = 'DTIRatio', bins = 10, hue = group, multiple = 'dodge')

# Interpret: Higher amount of defaults among those have higher DTI Ratio (for the most part)


# In[359]:


# Same but using our DTI_Class

sns.histplot(LoanData2, x = 'DTI_Classification', hue = group, multiple = 'dodge')

# Bad --> higher proportion of default 
# Good --> higher proportion of no default 

# maybe there is a link between DTI Ratio and the prob. of defaulting


# In[360]:


# DTI ratio classification for diff income groups 

pd.crosstab(LoanData2['DTI_Classification'], LoanData2['IncomeGroup'], normalize = 'columns')

# small differences between the income groups

# --> DTI not to do with abs income but debt to income ratio


# In[361]:


# --> look at average and median loan amounts across the diff income groups 

#Median 
MedInc = LoanData2['LoanAmount'].groupby(LoanData2['IncomeGroup']).median()
MeanInc = LoanData2['LoanAmount'].groupby(LoanData2['IncomeGroup']).mean()

print('Median Loan Amount by Income Group:', MedInc)
print('Mean Loan Amount by Income Group:', MeanInc)


# In[362]:


# Interpret: 

# Low Income group takes out larges loans 
# As Income Increases, mean amd median loan decreases 


# In[363]:


# Suggestion: Maybe younger individuals have worse DTI Ratio? 
# --> in reality, less income, more debt 

sns.histplot(LoanData2, x = 'DTI_Classification', hue = LoanData2['AgeGroup'], multiple = 'dodge')

# Both classes distributed in the same way across age groups


# In[364]:


sns.histplot(LoanData2, x = 'IncomeGroup', hue = group, multiple = 'dodge')


# # Categorical Variable Analysis

# In[365]:


# Has Mortage 

LoanData2.groupby(LoanData2['HasMortgage']).describe()


# In[366]:


# Distribution

sns.histplot(LoanData2['HasMortgage']) # there are significantly more applicants that do not hve mortgage


# In[367]:


# Is there a diff between the default groups? 

sns.histplot(LoanData2, x = 'HasMortgage', hue = LoanData2['Default'], multiple = 'dodge')

# Interpret: Signficantly more of those that do not have a mortgage default 

# --> those that have a mortgage may be a lot more stable financially 


# In[368]:


# Look at it numerically 

pd.crosstab(LoanData2['HasMortgage'], LoanData2['Default'], normalize = 'columns')

# 75% of those that default do not have a mortgage 
# Half of those that do not default have a mortgage


# In[369]:


# Are thee diff across age groups of nb of people with mortgages 

sns.histplot(LoanData2, x = 'HasMortgage', hue = LoanData2['AgeGroup'], multiple = 'dodge')

# Proportions the same in the two groups


# In[370]:


# What are the creditscores like for those with vs without mortgages? 

pd.crosstab(LoanData2['HasMortgage'], LoanData2['CreditScoreRating'], normalize = 'columns')

# Smaller amont of people with poor credit score have mortgage 
# BUT similar amount to those with very good credit score


# In[371]:


# Do same with DTI ratio

pd.crosstab(LoanData2['HasMortgage'], LoanData2['DTI_Classification'], normalize = 'columns') 

# slight diff between the two groups


# In[372]:


# Get dummies for IncomeGroup, CreditScoreRating,AgeGroup

Dummies2 =  pd.get_dummies(LoanData2[['AgeGroup', 'IncomeGroup', 'CreditScoreRating']])
LoanData2 = pd.concat([LoanData2, Dummies2], axis = 1)
LoanData2.head()


# In[373]:


# Drop object versions of categorical variables 

LoanData2 = LoanData2.drop(['AgeGroup', 'IncomeGroup', 'CreditScoreRating'], axis = 1)


# In[374]:


sns.histplot(LoanData2['CreditScoreRating'])


# # Evaluation Function

# In[ ]:


# import libraries to test performance 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# define function to evaluate the model 

def evaluate_preds(y_true, y_preds):
    
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels
    on a classification.
    
    
    """
    
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1:{f1:.2f}")
    
    return metric_dict
        
    


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from keras.models import Sequential
from keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix




# # Unbalanced Model

# In[ ]:


# Model before balancing the dataset 

LoanDataOriginal = LoanData.copy()
LoanDataOriginal.shape


# In[ ]:


# 
# Encodings
LoanDataOriginal.HasMortgage.replace(['Yes', 'No'], [1, 0], inplace=True)
LoanDataOriginal.HasDependents.replace(['Yes', 'No'], [1, 0], inplace=True)
LoanDataOriginal.HasCoSigner.replace(['Yes', 'No'], [1, 0], inplace=True)
LoanDataOriginal.DTI_Classification.replace(['Bad', 'Good'], [1,0], inplace = True)

# Convert categoricals
Dummies = pd.get_dummies(LoanDataOriginal[['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose', 
                                           'CreditScoreRating', 'AgeGroup', 'IncomeGroup']])

# Drop the originals
LoanDataOriginal.drop(['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose', 'CreditScoreRating', 
                       'AgeGroup', 'IncomeGroup'], axis=1, inplace=True)

# Concat
LoanDataOriginal = pd.concat([LoanDataOriginal, Dummies], axis=1)



# In[437]:


# building model on original data 

# Split data 

X = LoanDataOriginal.drop('Default', axis = 1)
y = LoanDataOriginal['Default']

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

# Logistic Regression Model 


# Initialise model 
log_reg = LogisticRegression()

# Fit model on training data 
log_reg.fit(X_train, y_train)


# Make baseline predictions

y_preds = log_reg.predict(X_test)

# Evaluate classifier on test set 
baseline_metrics = evaluate_preds(y_test, y_preds)
baseline_metrics


# In[438]:


# AUC / ROC
# Implement AUC_score, and ROC curve 

fpr1, tpr1, thresholds = roc_curve(y_test, y_preds)
roc_auc1 = auc(fpr1, tpr1)
display = metrics.RocCurveDisplay(fpr=fpr1, tpr = tpr1, roc_auc = roc_auc1)
display.plot()
plt.title('ROC Plot of Base Model on Balanced Dataset')


# In[ ]:


# encode dti class in loandata 2 

LoanData2.DTI_Classification.replace(['Bad', 'Good'], [1,0], inplace = True)


# # Base Model

# In[441]:


# Test model 

LoanData3 = LoanData2.copy()

# Split X, y
X = LoanData3.drop('Default', axis = 1)
y = LoanData3['Default']

# Split train-test
X_train30, X_test30, y_train30, y_test30 = train_test_split(X,y, stratify = y, test_size = 0.2, random_state = 123)



# Initialise 
log_reg2 = LogisticRegression()

# Fit model on training data 
log_reg2.fit(X_train30, y_train30)



# Make predictions

y_preds2 = log_reg2.predict(X_test30)


# Evaluate classifier on test set 
baselog = evaluate_preds(y_test30, y_preds2)
baselog


# In[ ]:


# Confusion Matrix 
confusion_matrix(y_test, y_preds2)


# In[442]:


# AUC, ROC
fpr2, tpr2, thresholds = roc_curve(y_test30, y_preds2)
roc_auc2 = auc(fpr2, tpr2)
display = metrics.RocCurveDisplay(fpr=fpr2, tpr = tpr2, roc_auc = roc_auc2)
display.plot()
plt.title('ROC Plot of Base Model on Balanced Dataset')


# # Model 2 - Standard Scaler

# In[ ]:


# Test


# Extract numerical variables
Num = LoanData2[['Age', 'Income', 'LoanAmount', 'CreditScore', 'YearsEmployed',
                  'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']]

# Store categorical attributes
Cat = LoanData2.drop(columns=Num.columns)

# Rebuild dataset
LoanData3 = pd.concat([Num, Cat], axis=1)

# Separate dependent and independent variables
X = LoanData3.drop('Default', axis=1)
y = LoanData3['Default']

# Split 
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, stratify=y, test_size=0.2, random_state=123)

# Initialize 
scaler = StandardScaler()

# Fit scaler
X_train_scaled = scaler.fit_transform(X_train2)

# apply scaler on test data
X_test_scaled = scaler.transform(X_test2)

# Initialize
log_reg3 = LogisticRegression(random_state=123)

# Fit model
log_reg3.fit(X_train_scaled, y_train2)

# predictions
y_preds3 = log_reg3.predict(X_test_scaled)

# Evaluate 
model2 = evaluate_preds(y_test2, y_preds3)
model2


# In[ ]:





# In[210]:


# results: accuracy: 93.33%, precision: 0.99, recall: 0.87, F1: 0.93

# assessment: significant increase in performance


# In[421]:


# Confusion Matrix 

confusion_matrix(y_test2, y_preds3)


# In[422]:


# AUC, ROC for model 2: 

fpr3, tpr3, thresholds = roc_curve(y_test2, y_preds3)
roc_auc3 = auc(fpr3, tpr3)
display = metrics.RocCurveDisplay(fpr=fpr3, tpr = tpr3, roc_auc = roc_auc3)

display.plot()
plt.title('ROC Plot of Logistic Regression With Standard Scaler')


# # Model 3 - Min Max Scaler Model

# In[423]:


from sklearn.preprocessing import MinMaxScaler


# numerical features
Num2 = LoanData2[['Age', 'Income', 'LoanAmount', 'CreditScore', 'YearsEmployed',
                   'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']]

# cat features
Cat2 = LoanData2.drop(columns=Num2.columns)

# Rebuild dataset
LoanData4 = pd.concat([Num2, Cat2], axis=1)

# splt x,y
X = LoanData4.drop('Default', axis=1)
y = LoanData4['Default']

# Split 
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.2, random_state=123)

# Initialize
scaler = MinMaxScaler()

# Fit scaler
X_train_scaled = scaler.fit_transform(X_train3)

# apply on test data
X_test_scaled = scaler.transform(X_test3)

# Initialize 
log_reg4 = LogisticRegression(random_state=123)

# Fit model
log_reg4.fit(X_train_scaled, y_train3)

# Predictions
y_preds4 = log_reg4.predict(X_test_scaled)

# Evaluate
model3 = evaluate_preds(y_test3, y_preds4)
model3


# In[424]:


# Confusion Matrix 
confusion_matrix(y_test3, y_preds4)


# In[425]:


# Implement AUC_score, and ROC curve 


fpr4, tpr4, thresholds = roc_curve(y_test3, y_preds4)
roc_auc4 = auc(fpr4, tpr4)
display = metrics.RocCurveDisplay(fpr=fpr4, tpr = tpr4, roc_auc = roc_auc4)
display.plot()


# In[426]:


# Compare 

# Create the plot
plt.figure()

# ROC 1
metrics.RocCurveDisplay(fpr=fpr4, tpr=tpr4, roc_auc=roc_auc4, estimator_name='Model Min Max Scaler').plot(alpha=0.5)

# ROC 2
metrics.RocCurveDisplay(fpr=fpr2, tpr=tpr2, roc_auc=roc_auc2, estimator_name='Base Model').plot(ax=plt.gca(), alpha=0.5)

# Roc3
metrics.RocCurveDisplay(fpr=fpr3, tpr=tpr3, roc_auc=roc_auc3, estimator_name='Model Standard Scaler').plot(ax=plt.gca(), alpha=0.5)

plt.title('ROC Curves of Base Model and Scaled Models')

# show
plt.show()



# In[427]:


# min max scaler performs slightly better on unseen data 


# In[428]:


# Odds ratios: 

# log odds 
log_odds = log_reg3.coef_[0]

# odds_ratios
odds_ratios = np.exp(log_odds)

# combine
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Log-Odds': log_odds,
    'Odds Ratio': odds_ratios
})

coefficients


# In[429]:


# Correlation analysis 

# coefficients have shown us the relationships between each attribute and the dependent variable,
# but what are the relationships between the independent variables like? 


# --> are any of the relationships between the numerical features linear? --> correlation analysis 

# CORRELATION ANALYSIS 

Num = LoanData2[['Income', 'Age', 'InterestRate', 'LoanAmount', 'CreditScore', 'NumCreditLines', 'LoanTerm']]

# extract numerical features --> we already have them stored in Num
Num.head()

# correlation
correlation = Num.corr().round(2)

# Plot 
import seaborn as sns 
sns.heatmap(correlation, cmap = 'RdBu_r', vmin = -1, vmax = 1, annot = True, linewidths=0.5)
plt.title('Correlation Matrix')



# In[430]:


# plot the relationships between these variables to confirm this: 

# Age-income 
sns.scatterplot(LoanData2, x = 'Age', y = 'Income')
plt.title('Scatterplot of Age-Income Relationship')


# In[431]:


# Age-CreditScore 
sns.scatterplot(LoanData2, x = 'Age', y = 'CreditScore')
plt.title('Scatterplot of Age-CreditScore Relationship')


# In[432]:


# Age-interest rate 

sns.scatterplot(LoanData2, x = 'Age', y = 'InterestRate')
plt.title('Relationship between Age and Interest Rate')


# # Mutual Information

# In[433]:


# Mutual Information

# chosen as it is one of the few methods that can be applied to mixed data types 

# what does it do? 

## -> tells us how on average, the change we see in one variable is related to the change in another variable

# import MI for discrete target variable 



# split data 

X = LoanData2.drop('Default', axis = 1)
y = LoanData2['Default']

# Split data 

X_train6, X_test6, y_train6, y_test6 = train_test_split(X, y, test_size = 0.2)

# Find Mutual Information 
mutual_info = mutual_info_classif(X_train6, y_train6)

# Restructure the mutual information values 
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train6.columns
mutual_info.sort_values(ascending = False)


# # Model 4 - Mutual Information 

# In[222]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


X = LoanData2[['YearsEmployed', 'DTIRatio', 'InterestRate', 'LoanTerm']]
y = LoanData2['Default']

# Split 
X_train7, X_test7, y_train7, y_test7 = train_test_split(X, y, stratify=y, test_size=0.2, random_state=123)

# Scale
scaler = MinMaxScaler()

# Fit scaler
X_train7_scaled = scaler.fit_transform(X_train7)

# Apply scaler
X_test7_scaled = scaler.transform(X_test7)

# Initialize
log_reg6 = LogisticRegression(random_state=123)

# Fit model
log_reg6.fit(X_train7_scaled, y_train7)

# Predictoins
y_preds6 = log_reg6.predict(X_test7_scaled)

# Evaluate 
model4 = evaluate_preds(y_test7, y_preds6)
model4


# # Model 5 - Categorical Features 

# In[394]:


Categorical= LoanData2[["Education_Bachelor's", "Education_High School", "Education_Master's",
       'Education_PhD', 'EmploymentType_Full-time', 'EmploymentType_Part-time',
       'EmploymentType_Self-employed', 'EmploymentType_Unemployed',
       'MaritalStatus_Divorced', 'MaritalStatus_Married',
       'MaritalStatus_Single', 'LoanPurpose_Auto', 'LoanPurpose_Business',
       'LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other', 'Default']]

X = Categorical.drop('Default', axis = 1)
y = Categorical['Default']

X_train8, X_test8, y_train8, y_test8 = train_test_split(X,y,test_size = 0.2)

# Initialise  
log_reg7 = LogisticRegression()

# Fit model
log_reg7.fit(X_train8, y_train8)

# Predictions
y_preds7 = log_reg7.predict(X_test8)

# Evaluate
model5 = evaluate_preds(y_test8, y_preds7)
model5


# In[395]:


# ROC AUC
fpr5, tpr5, thresholds = roc_curve(y_test8, y_preds7)
roc_auc5 = auc(fpr5, tpr5)

# Create the ROC curve display
display = metrics.RocCurveDisplay(fpr=fpr5, tpr=tpr5, roc_auc=roc_auc5)

# ROC
plt.figure()  
display.plot()  
plt.title('Model 5: Logistic Regression Using Subset Of Categorical Features')
plt.show()


# In[444]:


# Adding this ROC Curve to the previous ones 

# Create the plot
plt.figure(figsize=(10, 8))  # Adjust size for better readability

# Plot ROCs
metrics.RocCurveDisplay(fpr=fpr1, tpr=tpr1, roc_auc=roc_auc1, estimator_name='Unbalanced Model').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr2, tpr=tpr2, roc_auc=roc_auc2, estimator_name='Base Model').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr3, tpr=tpr3, roc_auc=roc_auc3, estimator_name='Model 3').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr4, tpr=tpr4, roc_auc=roc_auc4, estimator_name='Model 4').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr5, tpr=tpr5, roc_auc=roc_auc5, estimator_name='Model 5').plot(alpha=0.5, ax=plt.gca())

# info
plt.title('ROC Curves for All Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Show the plot
plt.show()


# # Recursive Feature Elimination

# In[396]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import pandas as pd

# extract num features
Num = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'YearsEmployed', 
       'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

# Separate 
X = LoanData2.drop('Default', axis=1)
y = LoanData2['Default']

# Split 
X_train9, X_test9, y_train9, y_test9 = train_test_split(X, y, test_size=0.2, random_state=1)

# Separate 
X_train9_num = X_train9[Num]
X_train9_cat = X_train9.drop(Num, axis=1)  # Categorical or non-numeric features

X_test9_num = X_test9[Num]
X_test9_cat = X_test9.drop(Num, axis=1)  # Categorical or non-numeric features

# Scale
scaler = MinMaxScaler()

# Fit scaler
X_train9_num_scaled = scaler.fit_transform(X_train9_num)

# Apply scaler
X_test9_num_scaled = scaler.transform(X_test9_num)

# Put as DF
X_train9_num_scaled = pd.DataFrame(X_train9_num_scaled, columns=Num, index=X_train9.index)
X_test9_num_scaled = pd.DataFrame(X_test9_num_scaled, columns=Num, index=X_test9.index)

# Merge
X_train9_scaled = pd.concat([X_train9_num_scaled, X_train9_cat], axis=1)
X_test9_scaled = pd.concat([X_test9_num_scaled, X_test9_cat], axis=1)

# Initialise 
log_reg8 = LogisticRegression()

# Initialise
rfe = RFE(estimator=log_reg8, n_features_to_select=22)

# Fit RFE
rfe.fit(X_train9_scaled, y_train9)

# Get necessary pieces
print("Selected Features:", rfe.support_)
print("Feature Ranking:", rfe.ranking_)

# Get list of features 
selected_features = X.columns[rfe.support_]
print("Selected Feature Names:", selected_features)


# In[398]:


# convert to df for interpretability 

pd.DataFrame(selected_features)


# # Model 6 - RFE First Half

# In[399]:


# RFE Model - First half

# aall of the features that are selected are categorical 


# build a model using these features 

RFE_features = LoanData2[["Education_Bachelor's", "Education_High School", "Education_Master's", 
                 "Education_PhD", "EmploymentType_Full-time", "EmploymentType_Part-time", 
                 "EmploymentType_Self-employed", "EmploymentType_Unemployed", 
                 "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single", 
                 "LoanPurpose_Auto", "LoanPurpose_Business", "LoanPurpose_Education", 
                 "LoanPurpose_Home", "LoanPurpose_Other", "IncomeGroup_Low", 
                 "IncomeGroup_Lower-Middle", "IncomeGroup_Middle", "IncomeGroup_Upper", 
                 "CreditScoreRating_Fair", "CreditScoreRating_Poor", 'Default']]

# Build the model 

X = RFE_features.drop('Default', axis = 1)
y = RFE_features['Default']

X_train10, X_test10, y_train10, y_test10 = train_test_split(X,y, test_size = 0.2)

# initialise
log_reg9 = LogisticRegression()

# Fit 
log_reg9.fit(X_train10, y_train10)

# Predictions

y_preds8 = log_reg9.predict(X_test10)

# Evaluate
model6 = evaluate_preds(y_test10, y_preds8)
model6


# In[ ]:


# Add to main plot 

# Add ROC curve to the plot with the previous ones 

# Adding this ROC Curve to the previous ones 

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc, roc_curve
import seaborn as sns

# ROC 1
fpr1, tpr1, thresholds1 = roc_curve(y_test2, y_preds3)
roc_auc1 = auc(fpr1, tpr1)

# ROC 2
fpr2, tpr2, thresholds2 = roc_curve(y_test3, y_preds4)
roc_auc2 = auc(fpr2, tpr2)

# ROC Base curve - base model
fpr3, tpr3, thresholds3 = roc_curve(y_test, y_preds2)
roc_auc3 = auc(fpr3, tpr3)

# Roc 4
fpr4, tpr4, thresholds4 = roc_curve(y_test7, y_preds6)
roc_auc4 = auc(fpr4, tpr4)

# ROC 5
fpr5, tpr5, thresholds5 = roc_curve(y_test8, y_preds7)
roc_auc5 = auc(fpr5, tpr5)

# ROC 6
fpr6, tpr6 , thresholds6 = roc_curve(y_test10, y_preds8)
roc_auc6 = auc(fpr6, tpr6)

# Create the plot
plt.figure(figsize=(10, 8))  # Adjust size for better readability

# Plot
metrics.RocCurveDisplay(fpr=fpr1, tpr=tpr1, roc_auc=roc_auc1, estimator_name='Model 2').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr2, tpr=tpr2, roc_auc=roc_auc2, estimator_name='Model 3').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr3, tpr=tpr3, roc_auc=roc_auc3, estimator_name='Base Model').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr4, tpr=tpr4, roc_auc=roc_auc4, estimator_name='Model 4').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr5, tpr=tpr5, roc_auc=roc_auc5, estimator_name='Model 5').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr6, tpr=tpr6, roc_auc=roc_auc6, estimator_name='Model 6').plot(alpha=0.5, ax=plt.gca())



# label
plt.title('ROC Curves for All Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Show the plot
plt.show()


# # Model 7 - Second Half

# In[400]:


Sample = LoanData2.drop(["Education_Bachelor's", "Education_High School", "Education_Master's", 
                 "Education_PhD", "EmploymentType_Full-time", "EmploymentType_Part-time", 
                 "EmploymentType_Self-employed", "EmploymentType_Unemployed", 
                 "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single", 
                 "LoanPurpose_Auto", "LoanPurpose_Business", "LoanPurpose_Education", 
                 "LoanPurpose_Home", "LoanPurpose_Other", "IncomeGroup_Low", 
                 "IncomeGroup_Lower-Middle", "IncomeGroup_Middle", "IncomeGroup_Upper", 
                 "CreditScoreRating_Fair", "CreditScoreRating_Poor"], axis = 1)


# In[401]:


Num = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'YearsEmployed', 
       'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

# Splt X, y
X = Sample.drop('Default', axis=1)
y = Sample['Default']  # Use the correct target column from `Sample`, not `Sample_scaled`

# Split 
X_train11, X_test11, y_train11, y_test11 = train_test_split(X, y, test_size=0.2, random_state=1)

# Separate
X_train_num = X_train11[Num]
X_train_cat = X_train11.drop(Num, axis=1)  # Categorical or non-numeric features

X_test_num = X_test11[Num]
X_test_cat = X_test11.drop(Num, axis=1)  # Categorical or non-numeric features

# Scaler
scaler = MinMaxScaler()

# transform
X_train_num_scaled = scaler.fit_transform(X_train_num)

# Apply scaler on test data
X_test_num_scaled = scaler.transform(X_test_num)

# Set as df
X_train_num_scaled = pd.DataFrame(X_train_num_scaled, columns=Num, index=X_train11.index)
X_test_num_scaled = pd.DataFrame(X_test_num_scaled, columns=Num, index=X_test11.index)

# Merge
X_train11_scaled = pd.concat([X_train_num_scaled, X_train_cat], axis=1)
X_test11_scaled = pd.concat([X_test_num_scaled, X_test_cat], axis=1)

# Initialize 
log_reg10 = LogisticRegression()

# Fit model
log_reg10.fit(X_train11_scaled, y_train11)

# Predictions
y_preds9 = log_reg10.predict(X_test11_scaled)

# Evaluate 
model7 = evaluate_preds(y_test11, y_preds9)
model7


# In[ ]:


# Confusion Matrix 

confusion_matrix(y_test11, y_preds9)


# In[402]:


# AUC, ROC for model 3: 

# AUC, ROC for model 2: 

# Implement AUC_score, and ROC curve 


fpr7, tpr7, thresholds = roc_curve(y_test11, y_preds9)
roc_auc7 = auc(fpr7, tpr7)
display = metrics.RocCurveDisplay(fpr=fpr7, tpr = tpr7, roc_auc = roc_auc7)

display.plot()


# In[234]:


# interpret: decrease in model performance compared to the other share of features 
# but the model still performs well overall 


# In[235]:


# suggests that these features are less significant 


# In[403]:


# Compare to the previous model 

# Sixth ROC curve 
fpr6, tpr6 , thresholds6 = roc_curve(y_test10, y_preds8)
roc_auc6 = auc(fpr6, tpr6)

fpr7, tpr7, thresholds = roc_curve(y_test11, y_preds9)
roc_auc7 = auc(fpr7, tpr7)

# Create the plot
plt.figure(figsize=(10, 8))  # Adjust size for better readability

metrics.RocCurveDisplay(fpr=fpr6, tpr=tpr6, roc_auc=roc_auc6, estimator_name='Model 6').plot(ax = plt.gca())
metrics.RocCurveDisplay(fpr=fpr7, tpr=tpr7, roc_auc=roc_auc7, estimator_name='Model 7').plot(ax = plt.gca())

plt.title('Model 6 vs Model 7')


# In[ ]:


# Add Model 6 to plot with all of the other models 

# Add to main plot 

# Add ROC curve to the plot with the previous ones 

# Adding this ROC Curve to the previous ones 



# ROC 1
fpr1, tpr1, thresholds1 = roc_curve(y_test2, y_preds3)
roc_auc1 = auc(fpr1, tpr1)

# ROC 2
fpr2, tpr2, thresholds2 = roc_curve(y_test3, y_preds4)
roc_auc2 = auc(fpr2, tpr2)

# ROC base
fpr3, tpr3, thresholds3 = roc_curve(y_test, y_preds2)
roc_auc3 = auc(fpr3, tpr3)

# ROC 4
fpr4, tpr4, thresholds4 = roc_curve(y_test7, y_preds6)
roc_auc4 = auc(fpr4, tpr4)

# ROC 5 
fpr5, tpr5, thresholds5 = roc_curve(y_test8, y_preds7)
roc_auc5 = auc(fpr5, tpr5)

# ROC 6 
fpr6, tpr6 , thresholds6 = roc_curve(y_test10, y_preds8)
roc_auc6 = auc(fpr6, tpr6)

# ROC 7
fpr7, tpr7, thresholds = roc_curve(y_test11, y_preds9)
roc_auc7 = auc(fpr7, tpr7)


# plot
plt.figure(figsize=(10, 8))  # Adjust size for better readability

# Plot
metrics.RocCurveDisplay(fpr=fpr1, tpr=tpr1, roc_auc=roc_auc1, estimator_name='Model 2').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr2, tpr=tpr2, roc_auc=roc_auc2, estimator_name='Model 3').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr3, tpr=tpr3, roc_auc=roc_auc3, estimator_name='Base Model').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr4, tpr=tpr4, roc_auc=roc_auc4, estimator_name='Model 4').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr5, tpr=tpr5, roc_auc=roc_auc5, estimator_name='Model 5').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr6, tpr=tpr6, roc_auc=roc_auc6, estimator_name='Model 6').plot(ax = plt.gca())
metrics.RocCurveDisplay(fpr=fpr7, tpr=tpr7, roc_auc=roc_auc7, estimator_name='Model 7').plot(ax = plt.gca())



# description
plt.title('ROC Curves for All Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')


plt.show()


# # New Models

# In[404]:


X = LoanData2.drop('Default', axis = 1)
y = LoanData2['Default']

#split 
X_train12, X_test12, y_train12, y_test12 = train_test_split(X,y, test_size = 0.2, stratify = y)


# In[405]:


from sklearn.tree import DecisionTreeClassifier


# initialise
clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 123)

# fit model
clf = clf.fit(X_train12, y_train12)

# predictions
tree_pred1 = clf.predict(X_test12)

# evaluate
basetree = evaluate_preds(y_test12, tree_pred1)
basetree


# In[406]:


# Confusion Matrix 

confusion_matrix(y_test2, tree_pred1)


# In[407]:


# AUC ROC


fpr8, tpr8, thresholds = roc_curve(y_test12, tree_pred1)
roc_auc8 = auc(fpr8, tpr8)
display = metrics.RocCurveDisplay(fpr=fpr8, tpr = tpr8, roc_auc = roc_auc8)

display.plot()


# In[240]:


# Add Model 6 to plot with all of the other models 

# Add to main plot 

# Add ROC curve to the plot with the previous ones 

# Adding this ROC Curve to the previous ones 



# Decision tree 1 - base tree 
fpr8, tpr8, thresholds = roc_curve(y_test2, tree_pred1)
roc_auc8 = auc(fpr8, tpr8)



# Create the plot
plt.figure(figsize=(10, 8))  # Adjust size for better readability

# Plot all ROC curves
metrics.RocCurveDisplay(fpr=fpr1, tpr=tpr1, roc_auc=roc_auc1, estimator_name='Model 1').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr2, tpr=tpr2, roc_auc=roc_auc2, estimator_name='Model 2').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr3, tpr=tpr3, roc_auc=roc_auc3, estimator_name='Base Model').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr4, tpr=tpr4, roc_auc=roc_auc4, estimator_name='Model 3').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr5, tpr=tpr5, roc_auc=roc_auc5, estimator_name='Model 4').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr6, tpr=tpr6, roc_auc=roc_auc6, estimator_name='Model 5').plot(ax = plt.gca())
metrics.RocCurveDisplay(fpr=fpr7, tpr=tpr7, roc_auc=roc_auc7, estimator_name='Model 6').plot(ax = plt.gca())
metrics.RocCurveDisplay(fpr=fpr8, tpr = tpr8, roc_auc = roc_auc8, estimator_name = 'DT Base Model').plot(ax=plt.gca())




# Add titles and labels
plt.title('ROC Curves for All Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Show the plot
plt.show()


# In[241]:


# interpret: this tree performs very well, but it is less good than the log model after standard scaler 


# In[242]:


# why? trees are very flexible --> may have traced the data rather thus affecting prediction 

# solution: try less features --> use the best subset of features selected so far and evaluate 


# # Decision Tree 2: Using best selected features so far 

# In[408]:


# almost all of the features that are selected are categorical 


# build a model using these features 

RFE_features = LoanData2[["Education_Bachelor's", "Education_High School", "Education_Master's", 
                 "Education_PhD", "EmploymentType_Full-time", "EmploymentType_Part-time", 
                 "EmploymentType_Self-employed", "EmploymentType_Unemployed", 
                 "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single", 
                 "LoanPurpose_Auto", "LoanPurpose_Business", "LoanPurpose_Education", 
                 "LoanPurpose_Home", "LoanPurpose_Other", "IncomeGroup_Low", 
                 "IncomeGroup_Lower-Middle", "IncomeGroup_Middle", "IncomeGroup_Upper", 
                 "CreditScoreRating_Fair", "CreditScoreRating_Poor", 'Default']]

# split X, y
X = RFE_features.drop('Default', axis = 1)
y = RFE_features['Default']

# train test split 

X_train12, X_test12, y_train12, y_test12 = train_test_split(X,y, test_size = 0.2, stratify = y)

# Initialise

clf2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 1)

# train decision tree 
clf2 = clf2.fit(X_train12, y_train12)

# make predictions 
tree_pred2 = clf2.predict(X_test12)

# evaluate performance 
tree2 = evaluate_preds(y_test12, tree_pred2)
tree2


# In[244]:


# AUC ROC

fpr9, tpr9, thresholds = roc_curve(y_test12, tree_pred2)
roc_auc9 = auc(fpr9, tpr9)
display = metrics.RocCurveDisplay(fpr=fpr9, tpr = tpr9, roc_auc = roc_auc9)

display.plot()


# In[245]:


# interpret: significant increase in performance 



# # Decision Tree 3: Other Subset of Features 

# In[409]:


# do we see a similar decrease as before when using the other half of the attributes? 

# scale the numeric features 

Sample = LoanData2.drop(["Education_Bachelor's", "Education_High School", "Education_Master's", 
                 "Education_PhD", "EmploymentType_Full-time", "EmploymentType_Part-time", 
                 "EmploymentType_Self-employed", "EmploymentType_Unemployed", 
                 "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single", 
                 "LoanPurpose_Auto", "LoanPurpose_Business", "LoanPurpose_Education", 
                 "LoanPurpose_Home", "LoanPurpose_Other", "IncomeGroup_Low", 
                 "IncomeGroup_Lower-Middle", "IncomeGroup_Middle", "IncomeGroup_Upper", 
                 "CreditScoreRating_Fair", "CreditScoreRating_Poor"], axis = 1)


# Split the data 

X = Sample.drop('Default', axis = 1)
y = Sample['Default']

# train-test split 
X_train13, X_test13, y_train13, y_test13 = train_test_split(X,y, test_size=0.2)

clf3 = DecisionTreeClassifier(criterion = 'entropy',random_state = 1)

# train decision tree 
clf3 = clf.fit(X_train13, y_train13)

# make predictions 
tree_pred3 = clf3.predict(X_test13)

# evaluate performance 
tree3 = evaluate_preds(y_test13, tree_pred3)
tree3


# In[410]:


confusion_matrix(y_test13, tree_pred3)


# In[411]:


# ROC-AUC
# AUC ROC

fpr10, tpr10, thresholds = roc_curve(y_test13, tree_pred3)
roc_auc10 = auc(fpr10, tpr10)
display = metrics.RocCurveDisplay(fpr=fpr10, tpr = tpr10, roc_auc = roc_auc10)

display.plot()


# In[412]:


# combine the two ROC curves into one plot 

metrics.RocCurveDisplay(fpr=fpr9, tpr = tpr9, roc_auc = roc_auc9, estimator_name = 'DT 2').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr10, tpr = tpr10, roc_auc = roc_auc10, estimator_name = 'DT 3').plot(alpha=0.5, ax=plt.gca())

plt.title('ROC Curves of Decision Trees 2 and 3')


# # Decision Tree 4 

# In[413]:


# what if we added more of the standard features to the first set of features? 

# all of the features in RFE list are cat --> add numerical ones for comparison  


# build a model using these features 

RFE_features = LoanData2[['Age', 'LoanAmount', 'YearsEmployed', 'InterestRate', 'LoanTerm', 'Income', 'DTIRatio', 
                          'CreditScore', "Education_Bachelor's", "Education_High School", "Education_Master's", 
                 "Education_PhD", "EmploymentType_Full-time", "EmploymentType_Part-time", 
                 "EmploymentType_Self-employed", "EmploymentType_Unemployed", 
                 "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single", 
                 "LoanPurpose_Auto", "LoanPurpose_Business", "LoanPurpose_Education", 
                 "LoanPurpose_Home", "LoanPurpose_Other", "IncomeGroup_Low", 
                 "IncomeGroup_Lower-Middle", "IncomeGroup_Middle", "IncomeGroup_Upper", 
                 "CreditScoreRating_Fair", "CreditScoreRating_Poor", 'Default']]


# split X, y
X = RFE_features.drop('Default', axis = 1)
y = RFE_features['Default']

# train test split 

X_train14, X_test14, y_train14, y_test14 = train_test_split(X,y, test_size = 0.2, stratify = y, random_state = 123)


clf4 = DecisionTreeClassifier(criterion = 'entropy', random_state = 1)

# fit
clf4 = clf4.fit(X_train14, y_train14)

# predictions 
tree_pred4 = clf4.predict(X_test14)

# evaluate 
tree4 = evaluate_preds(y_test14, tree_pred4)
tree4


# In[250]:


# AUC ROC

fpr11, tpr11, thresholds = roc_curve(y_test14, tree_pred4)
roc_auc11 = auc(fpr11, tpr11)
display = metrics.RocCurveDisplay(fpr=fpr11, tpr = tpr11, roc_auc = roc_auc11)

display.plot()


# In[251]:


# Combine all of the tree models into one plot 

# combine the two ROC curves into one plot 

metrics.RocCurveDisplay(fpr=fpr9, tpr = tpr9, roc_auc = roc_auc9, estimator_name = 'DT 2').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr10, tpr = tpr10, roc_auc = roc_auc10, estimator_name = 'DT 3').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr11, tpr = tpr11, roc_auc = roc_auc11, estimator_name = 'DT 4').plot(alpha=0.5, ax=plt.gca())

plt.title('ROC Curves of Decision Trees 2,3 and 4')





# # Random Forests 

# In[414]:


# set up RF
from sklearn.ensemble import RandomForestClassifier


# no feature selection 
X = LoanData2.drop('Default', axis = 1)
y = LoanData2['Default']


# train-test split 
X_train15, X_test15, y_train15, y_test15 = train_test_split(X,y, test_size = 0.2)

# initialise
rf1 = RandomForestClassifier(random_state = 123)

# fit 
rf1.fit(X_train15, y_train15)

# predictions
rfbase = rf1.predict(X_test15)


# evaluate 
baseforest = evaluate_preds(y_test15, rfbase)
baseforest


# In[415]:


# ROC

fpr12, tpr12, thresholds = roc_curve(y_test15, rfbase)
roc_auc12 = auc(fpr12, tpr12)
display = metrics.RocCurveDisplay(fpr=fpr12, tpr = tpr12, roc_auc = roc_auc12)
display.plot()
plt.title('Random Forest Base Model')


# # Random Forest 2

# In[416]:


# Random Forest 2 

# Building Random Forest only using first share of features 

# build a model using these features 

RFE_features = LoanData2[["Education_Bachelor's", "Education_High School", "Education_Master's", 
                 "Education_PhD", "EmploymentType_Full-time", "EmploymentType_Part-time", 
                 "EmploymentType_Self-employed", "EmploymentType_Unemployed", 
                 "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single", 
                 "LoanPurpose_Auto", "LoanPurpose_Business", "LoanPurpose_Education", 
                 "LoanPurpose_Home", "LoanPurpose_Other", "IncomeGroup_Low", 
                 "IncomeGroup_Lower-Middle", "IncomeGroup_Middle", "IncomeGroup_Upper", 
                 "CreditScoreRating_Fair", "CreditScoreRating_Poor", 'Default']]


# split X, y
X = RFE_features.drop('Default', axis = 1)
y = RFE_features['Default']

# train test split 

X_train16, X_test16, y_train16, y_test16 = train_test_split(X,y, test_size = 0.2)


# initialise
rf2 = RandomForestClassifier(random_state = 123)

# train forest 
rf2 = rf2.fit(X_train16, y_train16)

# predictions 
rf2model = rf2.predict(X_test16)

# evaluate
forest2 = evaluate_preds(y_test16, rf2model)
forest2


# In[417]:


# Performance is about the same 

# ROC - AUc


fpr13, tpr13, thresholds = roc_curve(y_test16, rf2model)
roc_auc13 = auc(fpr13, tpr13)

metrics.RocCurveDisplay(fpr=fpr12, tpr = tpr12, roc_auc = roc_auc12, estimator_name = 'RF Base').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr13, tpr=tpr13, roc_auc = roc_auc13, estimator_name = 'RF 2').plot(alpha=0.5, ax=plt.gca())
plt.title('Random Forest Base and 2')


# In[253]:


# Interpret: reducing the number of features maintains performance like with logistic regression 


# # Random Forest 3 

# In[418]:


# Random Forest with second set of features 

# do we see a similar decrease as before when using the other half of the attributes? 

Sample = LoanData2.drop(["Education_Bachelor's", "Education_High School", "Education_Master's", 
                 "Education_PhD", "EmploymentType_Full-time", "EmploymentType_Part-time", 
                 "EmploymentType_Self-employed", "EmploymentType_Unemployed", 
                 "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single", 
                 "LoanPurpose_Auto", "LoanPurpose_Business", "LoanPurpose_Education", 
                 "LoanPurpose_Home", "LoanPurpose_Other", "IncomeGroup_Low", 
                 "IncomeGroup_Lower-Middle", "IncomeGroup_Middle", "IncomeGroup_Upper", 
                 "CreditScoreRating_Fair", "CreditScoreRating_Poor"], axis = 1)


# Split the data 

X = Sample.drop('Default', axis = 1)
y = Sample['Default']

# train-test split 
X_train17, X_test17, y_train17, y_test17 = train_test_split(X,y, test_size=0.2)

# initialise

rf3 = RandomForestClassifier(random_state = 123)

# train decision tree 
rf3 = rf3.fit(X_train17, y_train17)

# predict
rf3model = rf3.predict(X_test17)

# evaluate
forest3 = evaluate_preds(y_test17, rf3model)
forest3


# In[ ]:


# decrease in performance like with th previous models 



# In[419]:


# Add this RF to the plot with the other two for comparison 

# Add to ROC-AUC 

fpr14, tpr14, thresholds = roc_curve(y_test17, rf3model)
roc_auc14 = auc(fpr14, tpr14)

metrics.RocCurveDisplay(fpr=fpr12, tpr = tpr12, roc_auc = roc_auc12, estimator_name = 'RF Base').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr13, tpr=tpr13, roc_auc = roc_auc13, estimator_name = 'RF 2').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr14, tpr=tpr14, roc_auc = roc_auc14, estimator_name = 'RF 3').plot(alpha = 0.5, ax=plt.gca())
plt.title('Random Forests')




# # Neural Networks 

# In[ ]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

# Base Model
# Split data 
X = LoanData2.drop('Default', axis=1)
y = LoanData2['Default']

# Split
X_train18, X_test18, y_train18, y_test18 = train_test_split(X, y, test_size=0.2, random_state=42)

# hidden units
hidden_units = 40

# Initialize 
NnBase = Sequential()
NnBase.add(Input(shape=(X_train18.shape[1],)) )  
NnBase.add(Dense(hidden_units, activation='relu'))  
NnBase.add(Dense(1, activation='sigmoid'))  

# compile 
NnBase.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
NnBase.summary()

# train 
hist = NnBase.fit(X_train18, y_train18, validation_data=(X_test18, y_test18),
                  epochs=20, batch_size=100, verbose=1)

# predictions
npredsBase = NnBase.predict(X_test18) > 0.5


evaluate_preds(y_test18, npredsBase)


# # Neural Network 2 - MinMaxScaler

# In[ ]:


# Split 
X = LoanData2.drop('Default', axis=1)
y = LoanData2['Default']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# num features
num_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'YearsEmployed',
                'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

# scale
scaler = MinMaxScaler()

# apply scaler
X_train_scaled = X_train.copy()
X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])

# apply scaler pt 2 
X_test_scaled = X_test.copy()
X_test_scaled[num_features] = scaler.transform(X_test[num_features])



# initialise
model = Sequential()
model.add(Dense(40, activation='relu', input_shape=(X_train_scaled.shape[1],)))  # Input layer
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# compile
model.compile(loss='binary_crossentropy', metrics=['accuracy'])

# get summary of the model 
model.summary()

# train
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
                    epochs=20, batch_size=100)

# Make predictions
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)  

evaluate_preds(y_test, y_pred)


# # Neural Network 3 - RFE 

# In[ ]:


#what if we drop the second set of features  

# build a model using these features 

RFE_features = LoanData2[["Education_Bachelor's", "Education_High School", "Education_Master's", 
                 "Education_PhD", "EmploymentType_Full-time", "EmploymentType_Part-time", 
                 "EmploymentType_Self-employed", "EmploymentType_Unemployed", 
                 "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single", 
                 "LoanPurpose_Auto", "LoanPurpose_Business", "LoanPurpose_Education", 
                 "LoanPurpose_Home", "LoanPurpose_Other", "IncomeGroup_Low", 
                 "IncomeGroup_Lower-Middle", "IncomeGroup_Middle", "IncomeGroup_Upper", 
                 "CreditScoreRating_Fair", "CreditScoreRating_Poor", 'Default']]





# split X, y
X = RFE_features.drop('Default', axis = 1)
y = RFE_features['Default']

X_train17, X_test17, y_train17, y_test17 = train_test_split(X,y, test_size = 0.2)

# hidden units 
hidden_units = 40

Nn2 = Sequential()
Nn2.add(Dense(hidden_units, activation = 'relu', input_dim = 22))
Nn2.add(Dense(1, activation='sigmoid'))  # For binary classification
Nn2.compile(loss = 'binary_crossentropy', optimizer = 'adam')
Nn2.summary()



# train the model 

hist = Nn2.fit(X_train17, y_train17, validation_data = (X_test17, y_test17),
                 epochs  = 20, batch_size = 100)

npreds2 = Nn2.predict(X_test17) > 0.5

# evaluate 

evaluate_preds(y_test17, npreds2) # more or less the same as log reg


# In[ ]:


# Plotting the two neural networks 

# Add this RF to the plot with the other two for comparison 

# interpret: slight increase in performance 

# ROC

fpr, tpr, thresholds = roc_curve(y_test16, npredsBase)
roc_auc = auc(fpr, tpr)

fpr2, tpr2, thresholds = roc_curve(y_test17, npreds2)
roc_auc2 = auc(fpr2, tpr2)



metrics.RocCurveDisplay(fpr=fpr, tpr = tpr, roc_auc = roc_auc, estimator_name = 'NN Base').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr2, tpr=tpr2, roc_auc = roc_auc2, estimator_name = 'NN 2').plot(alpha=0.5, ax=plt.gca())
plt.title('Neural Networks')




# In[ ]:


# ROC Plot of all of the models 

# Add Model 6 to plot with all of the other models 

# Add to main plot 

# Add ROC curve to the plot with the previous ones 

# Adding this ROC Curve to the previous ones 

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc, roc_curve
import seaborn as sns

# First ROC curve
fpr1, tpr1, thresholds1 = roc_curve(y_test2, y_preds3)
roc_auc1 = auc(fpr1, tpr1)

# Second ROC curve
fpr2, tpr2, thresholds2 = roc_curve(y_test3, y_preds4)
roc_auc2 = auc(fpr2, tpr2)

# Third ROC curve - base model
fpr3, tpr3, thresholds3 = roc_curve(y_test, y_preds2)
roc_auc3 = auc(fpr3, tpr3)

# Fourth ROC curve
fpr4, tpr4, thresholds4 = roc_curve(y_test7, y_preds6)
roc_auc4 = auc(fpr4, tpr4)

# Fifth ROC curve 
fpr5, tpr5, thresholds5 = roc_curve(y_test8, y_preds7)
roc_auc5 = auc(fpr5, tpr5)

# Sixth ROC curve 
fpr6, tpr6 , thresholds6 = roc_curve(y_test9, y_preds8)
roc_auc6 = auc(fpr6, tpr6)

# Seventh ROC Curve 
fpr7, tpr7, thresholds = roc_curve(y_test10, y_preds9)
roc_auc7 = auc(fpr7, tpr7)

# Decision tree 1 - base tree 
fpr8, tpr8, thresholds = roc_curve(y_test2, tree_pred1)
roc_auc8 = auc(fpr8, tpr8)

# DT 2
fpr9, tpr9, thresholds = roc_curve(y_test10, tree_pred2)
roc_auc9 = auc(fpr9, tpr9)

#DT 3
fpr10, tpr10, thresholds = roc_curve(y_test11, tree_pred3)
roc_auc10 = auc(fpr10, tpr10)


# DT 4
fpr11, tpr11, thresholds = roc_curve(y_test12, tree_pred4)
roc_auc11 = auc(fpr11, tpr11)


#RF Base
fpr12, tpr12, thresholds = roc_curve(y_test13, rfbase)
roc_auc12 = auc(fpr12, tpr12)

# RF 2
fpr13, tpr13, thresholds = roc_curve(y_test14, rf2model)
roc_auc13 = auc(fpr13, tpr13)

#RF 3
fpr14, tpr14, thresholds = roc_curve(y_test15, rf3model)
roc_auc14 = auc(fpr14, tpr14)




# Create the plot
plt.figure(figsize=(10, 8))  # Adjust size for better readability

# Plot all ROC curves
metrics.RocCurveDisplay(fpr=fpr1, tpr=tpr1, roc_auc=roc_auc1, estimator_name='Model 1').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr2, tpr=tpr2, roc_auc=roc_auc2, estimator_name='Model 2').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr3, tpr=tpr3, roc_auc=roc_auc3, estimator_name='Base Model').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr4, tpr=tpr4, roc_auc=roc_auc4, estimator_name='Model 3').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr5, tpr=tpr5, roc_auc=roc_auc5, estimator_name='Model 4').plot(alpha=0.5, ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr6, tpr=tpr6, roc_auc=roc_auc6, estimator_name='Model 5').plot(ax = plt.gca())
metrics.RocCurveDisplay(fpr=fpr7, tpr=tpr7, roc_auc=roc_auc7, estimator_name='Model 6').plot(ax = plt.gca())
metrics.RocCurveDisplay(fpr=fpr8, tpr = tpr8, roc_auc = roc_auc8, estimator_name = 'DT Base Model').plot(ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr9, tpr = tpr9, roc_auc = roc_auc9, estimator_name = 'DT 2').plot(ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr10, tpr = tpr10, roc_auc = roc_auc10, estimator_name = 'DT 3').plot(ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr11, tpr = tpr11, roc_auc = roc_auc11, estimator_name = 'DT 4').plot(ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr12, tpr = tpr12, roc_auc = roc_auc12, estimator_name = 'RF Base').plot(ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr13, tpr = tpr13, roc_auc = roc_auc13, estimator_name = 'RF 2').plot(ax=plt.gca())
metrics.RocCurveDisplay(fpr=fpr14, tpr = tpr14, roc_auc = roc_auc14, estimator_name = 'RF 3').plot(ax=plt.gca())




# Add titles and labels
plt.title('ROC Curves for All Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Show the plot
plt.show()

