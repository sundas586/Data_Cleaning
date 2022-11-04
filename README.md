# Data_Cleaning

Good data is the fuel that powers Machine Learning and Artificial Intelligence.<br />
Cleaning data ensures the quality,
In this repository, I will be putting some of my data cleaning practice codes.<br/>
-----------------------------------------------------------------------------------------<br/>

## Messy Data :

Observations of the data usually a row of the dataset. Think if a row is not clean, we are misrepresenting to our model, the relationship between our features and our targets, messy data can lead to "garbage-in, garbage-out" effect.

- Ensure that our data is **clean**.
- There must be **sufficient relevant data**.
- Avoid **data duplicates**, that can bring extra weight to observations/unnecessary noise to model.
- Avoid **inconsistent text** and typos (wrong spelling, extra spaces, the capitalized/non-capitalized letters. This will all lead to the same feature, being categorized                                          as different values, even though they should be categorized as the same.)
- **Missing data**. (Null entries, and why they are null).
- **Outliers** (that can skew a feature disproportionately and make it difficult to find the true underlying model.)
- **Data sourcing issues** (Trouble bringing in data from multiple systems, or working with different database types, or trying to wrangle and combine data coming from                               on-premises versus on-the-cloud or many others, makes it difficult to combine them).

## Policies for missing data :

- **Remove** row that has missing value, we can simply  (but if alot of rows have missing values,removing them we will end up biasing our dataset for the model).
- **Impute** them (means that we would be replacing our null values with either the mean or the median, or even a more complex estimation of the value. The pro to this                      is that we don't lose full rows or columns that may be important to our model, con is that we add another level of uncertainty to our model).
- **Mask** the missing data (assume that all of our missing values are their own category. This would be under the assumption that missing values are actually indicative                              of useful information. If we imagine a survey given by phone, and the last question is repeatedly blank, that may mean that the person hung                              up before getting to that last question, or we took a survey and people did not answer on how many children they have, is null because they                              actually dont have any children, so we mask all null values as zero children)

