[Kaggle data cleaning course and codes](https://www.kaggle.com/learn/data-cleaning)

# Data_Cleaning

Good data is the fuel that powers Machine Learning and Artificial Intelligence.<br />
Cleaning data ensures the quality,
In this repository, I will be putting some of my data cleaning practice codes.<br/>
-----------------------------------------------------------------------------------------<br/>

## Messy Data :

Observations of the data usually a row of the dataset. Think if a row is not clean, we are misrepresenting to our model, the relationship between our features and our targets, messy data can lead to "garbage-in, garbage-out" effect.

To ensure that our data is clean :
- **Skewed data** , it can be handle by transforming it into normally distributed data by doing log transformation or box cox method or square root transformation, [here is an article](https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45)
- Avoid **data duplicates**, that can bring extra weight to observations/unnecessary noise to model.
- Avoid **inconsistent text** and typos (wrong spelling, extra spaces, the capitalized/non-capitalized letters. This will all lead to the same feature, being   
                                        categorized as different values, even though they should be categorized as the same.)
- **Missing data**. (Null entries, and why they are null).
- **Outliers** (that can skew a feature disproportionately and make it difficult to find the true underlying model.)
- **Data sourcing issues** (Trouble bringing in data from multiple systems, or working with different database types, or trying to wrangle and combine data coming from                               on-premises versus on-the-cloud or many others, makes it difficult to combine them).
- **Scaling and Normalization** (in scaling, you change the range of your data, in normalization, you change the shape of the distribution of your data).
- **Character Encodings** (Avoid Uni-code-Decode-Errors when loading CSV files).
- There must be **sufficient relevant data**.

## Policies for missing data :

- **Remove** row that has missing value, we can simply  (but if alot of rows have missing values,removing them we will end up biasing our dataset for the model).
- **Impute** them (means that we would be replacing our null values with either the mean or the median, or even a more complex estimation of the value. The pro to this                      is that we don't lose full rows or columns that may be important to our model, con is that we add another level of uncertainty to our model).
- **Mask** the missing data (assume that all of our missing values are their own category. This would be under the assumption that missing values are actually                                      indicative of useful information. If we imagine a survey given by phone, and the last question is repeatedly blank, that may mean that the                              person hung up before getting to that last question, or we took a survey and people did not answer on how many children they have, is null                              because they actually dont have any children, so we mask all null values as zero children).


## Policies for Outliers/ Residuals : 

 The difference between the actual value and the predicted value given your model is an outlier, and they are going to represent model failure.
 
 ![residual](https://user-images.githubusercontent.com/33677647/200022165-24d6aee4-3932-4a35-912e-29ae07125d22.png)

- **Standardized th residual**, which is, it's going to take our residual and divide it by the standard error. The idea being here, if you have outcome variables between 10 million and 100 million versus between zero and five, being off by four means much different thing given those different outcome ranges. So we want to standardized it according to those ranges.

- **Deleted residuals**, What you can do here is you remove that observation from the dataframe. Once you do that, you see what the new model will predict and see if there's a big difference between the original model and the model with this deleted observation.

- **Studentized residuals**, This is essentially just going to take our deleted residuals and standardize them as we saw in our first standardized residual discussion. So it'll take out one of those observations. Once you take out that observation, you see the effect on the model and you standardize that according to the range of that model.
 
- **Remove them altogether** The pro to this is that you no longer need to worry about their effects, but you may have lost, not only an important value, but the entire row related to that value. You can assign a different value to the outlier.
 
 
 
 
 
 
 
