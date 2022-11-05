import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import scikit-learn as skln
import scipy as sp
import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import norm
from scipy import stats

##### Fetch DataSet ###

## Install a pip package in the current Jupyter kernel
## run the below lines only once to download, then commment then

import sys
!{sys.executable} -m pip install opendatasets 
import opendatasets as od
URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data1.tsv'
od.download(URL)

##### Load in the Ames Housing Data ###

# I save the downloaded .tsv file to .csv file
housing = pd.read_csv('Ames_Housing_Data1.csv')
housing.head()
housing["SalePrice"].describe()

##From the above analysis, it is important to note that the minimum value is greater than 0.
##Also, there is a big difference between the minimum value and the 25th percentile.
##It is bigger than the 75th percentile and the maximum value.
##This means that our data might not be normally distributed (an important assumption for linear regression analysis),
##so will check for normality in the Log Transform section.

housing["Sale Condition"].value_counts()

## Before proceeding with the data cleaning, 
## it is useful to establish a correlation between the response variable (in our case the sale price) and other predictor variables
## as some of them might not have any major impact in determining the price of the house and will not be used in the analysis. 
## There are many ways to discover correlation between the target variable and the rest of the features. 
## Building pair plots, scatter plots, heat maps, and a correlation matrixes are the most common ones. 
## Below, we will use the corr() function to list the top features based on the pearson correlation coefficient 
## (measures how closely two sequences of numbers are correlated).
## Correlation coefficient can only be calculated on the numerical attributes (floats and integers),
## therefore, only the numeric attributes will be selected.

# 
    
## From Pearsons Correlation Coefficients and pair plots, 
## we can draw some conclusions about the features that are most strongly correlated to the 'SalePrice'.
## They are: 'Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area', and others.    

###### LOG TRANSFORMATION ###

## In this section, we are going to inspect whether our 'SalePrice' data are normally distributed.
## The assumption of the normal distribution must be met in order to perform any type of regression analysis. 
## There are several ways to check for this assumption, however here, we will use the visual method,
## by plotting the 'SalePrice' distribution using the distplot() function from the seaborn library.

##Normal dist. is the most important probability distribution in statistics 
##because it accurately describes the distribution of values for many natural phenomena.

sp_untransformed = sns.distplot(housing['SalePrice'])

## As the plot shows, our 'SalePrice' deviates from the normal distribution. 
## It has a longer tail to the right, so we call it a positive skew.
## In statistics skewness is a measure of asymmetry of the distribution. 
## In addition to skewness, there is also a kurtosis, 
## parameter which refers to the pointedness of a peak in the distribution curve.
## Both skewness and kurtosis are frequently used together to characterize the distribution of data.

## Here, we can simply use the skew() function to calculate our skewness level of the SalePrice.

print("Skewness: %f" % housing['SalePrice'].skew())

## The range of skewness for a fairly symmetrical bell curve distribution is between -0.5 and 0.5; 
## moderate skewness is -0.5 to -1.0 and 0.5 to 1.0; and highly skewed distribution is < -1.0 and > 1.0. 
## In our case, we have ~1.7, so it is considered highly skewed data.

## Now, we can try to transform our data, so it looks more normally distributed.
## We can use the np.log() function from the numpy library to perform log transform.
## This documentation contains more information about the numpy log transform.

 
log_transformed = np.log(housing['SalePrice'])
sp_transformed = sns.distplot(log_transformed)
print("Skewness: %f" % (log_transformed).skew())

## As we can see, the log method transformed the 'SalePrice' distribution into a more symmetrical bell curve
## and the skewness level now is -0.01, well within the range.
## There are other ways to correct for skewness of the data. 
## For example, Square Root Transform (np.sqrt) and the Box-Cox Transform (stats.boxcox from the scipy stats library).

## Below line of code visually inspect the 'Lot Area' feature.
## If there is any skewness present, apply log transform to make it more normally distributed.

_untransformed = sns.distplot(housing['Lot Area'])
print("Skewness: %f" % housing['Lot Area'].skew())

## Skewness in lot area is 12.77%, lets perform log transformation

la_log = np.log(housing['Lot Area'])
print("Skewness: %f" % la_log.skew())
la_plot = sns.distplot(la_log)

##Handling the Duplicates
##We will use pandas duplicated() function and search by the 'PID' column,
##which contains a unique index number for each entry

duplicate = housing[housing.duplicated(['PID'])]
duplicate

## As we can see, there is one duplicate row in this dataset.
## To remove it, we can use pandas drop_duplicates() function. 
## By default, it removes all duplicate rows based on all the columns.

dup_removed = housing.drop_duplicates()
dup_removed 

## An alternative way to check if there are any duplicated Indexes in our dataset is using `index.is_unique` function.

housing.index.is_unique

##try to remove duplicates on a specific column by setting the subset equal to the column that contains the duplicate,
#such as 'Order'.

removed_sub = housing.drop_duplicates(subset=['Order'])

##### MISSING VALUES ###

## For easier detection of missing values, pandas provides the isna(), isnull(), and notna() functions.
## To summarize all the missing values in our dataset, we will use isnull() function.
## Then, we will add them all up, by using sum() function, sort them with sort_values() function,
## and plot the first 20 columns (as the majority of our missing values fall within first 20 columns),
## using the bar plot function from the matplotlib library.

total = housing.isnull().sum().sort_values(ascending=False)
total_select = total.head(20)
total_select.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Values", fontsize = 20)

#There are several options for dealing with missing values. 
# We will use 'Lot Frontage' feature to analyze for missing values.

# 1_ We can drop the missing values, using dropna() method, all the rows, containing null values in 'Lot Frontage' be removed.
housing.dropna(subset=["Lot Frontage"])

# 2_ We can drop the whole attribute (column), that contains missing values, using the drop() method.
housing.drop("Lot Frontage", axis=1)

# 3_We can replace the missing values(null) with zero/ mean/ median, etc, using fillna() method.
median = housing["Lot Frontage"].median()
median
housing["Lot Frontage"].fillna(median, inplace = True)
housing.tail()

# look at 'Mas Vnr Area' feature and replace the missing values with the mean value of that column.
    
mean = housing["Mas Vnr Area"].mean()
housing["Mas Vnr Area"].fillna(mean, inplace = True)   

###### Feature Scaling ###

# One of the most important transformations we need to apply to our data is feature scaling.
# There are two common ways to get all attributes to have the same scale: min-max scaling and standardization.
# Min-max scaling (or normalization) is the simplest: values are shifted and rescaled so they end up ranging from 0 to 1.
# This is done by subtracting the min value and dividing by the max minus min.
# Standardization is different: first it subtracts the mean value (so standardized values always have a zero mean),
# and then it divides by the standard deviation, so that the resulting distribution has unit variance.
# Scikit-learn library provides MinMaxScaler for normalization and StandardScaler for standardization needs. For more information on scikit-learn MinMaxScaler and StandardScaler please visit their respective documentation websites.

## First, we will normalize our data.

norm_data = MinMaxScaler().fit_transform(hous_num)
#norm_data

## Note the data is now a ndarray
## we can also standardize our data.

scaled_data = StandardScaler().fit_transform(hous_num)
#scaled_data

##use StandardScaler() and fit_transform() functions to standardize the 'SalePrice' feature only.

scaled_sprice = StandardScaler().fit_transform(housing['SalePrice'][:,np.newaxis]) 
#scaled_sprice





