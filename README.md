# Ex.No.1 Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries

Importing the dataset

Taking care of missing data

Encoding categorical data

Normalizing the data

Splitting the data into test and train

## PROGRAM:
```
Name: SWATHI D
Reg NO: 212222230154
```
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df

#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

#Checking for null values
df.isnull().sum()

#Checking for duplicate values
df.duplicated()

#Describing the dataset
df.describe()

#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:
## DATASET:
![image](https://github.com/swathidd/Ex.No.1---Data-Preprocessing/assets/121300272/dfa1cb07-5f43-4d20-b63c-73df8da4e045)

## Dropping unwanted features:
![image](https://github.com/swathidd/Ex.No.1---Data-Preprocessing/assets/121300272/f684a25a-94e1-4cf6-84ab-4ba005c38ac0)

## Checking for duplication:
![image](https://github.com/swathidd/Ex.No.1---Data-Preprocessing/assets/121300272/fb3d28a9-292e-4f53-af93-d6704f574636)

## Describing the dataset:
![image](https://github.com/swathidd/Ex.No.1---Data-Preprocessing/assets/121300272/1acef18c-73cf-4040-8c8c-067b84737ab1)

## Scaling the values:
![image](https://github.com/swathidd/Ex.No.1---Data-Preprocessing/assets/121300272/f98fa41d-64dc-437b-a5d5-788a44453f90)

## X Features:
![image](https://github.com/swathidd/Ex.No.1---Data-Preprocessing/assets/121300272/617b8097-9def-4c07-a7ec-58d55640d3c7)

## Y Features:
![image](https://github.com/swathidd/Ex.No.1---Data-Preprocessing/assets/121300272/8b626659-de6f-4ccf-8c41-6ab13c884941)

## Splitting the training and testing dataset:
![image](https://github.com/swathidd/Ex.No.1---Data-Preprocessing/assets/121300272/44d84317-d8b8-4840-bcbd-35aaf3d67352)


## RESULT
Thus we have successfully performed Data preprocessing in a data set downloaded from Kaggle
