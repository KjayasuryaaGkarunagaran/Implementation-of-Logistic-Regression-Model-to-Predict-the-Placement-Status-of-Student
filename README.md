# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1: start the program

step 2: Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).

step 3: Split the data into training and testing sets, then scale (standardize) both the input features and target variables.

step 4: Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.

step 4: Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.

step 5: End the program
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Stat
Developed by:jayasuryaa k
Register Number: 212222040060
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("C:/Users/admin/Downloads/Placement_Data (1).csv")
dataset
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
20/09/2024, 13:23 ligneshwar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
https://github.com/ligneshwar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student 1/2
Thus the program to implement the the Logistic Regression Model to Predict the Placement
Status of Student is written and verified using python programming
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:,-1].values
Y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
return 1/(1+np.exp(-z))
def loss(theta,X,y):
h=sigmoid(dot(theta))
return -np.sum(y*np.log(h)+(1-y)*log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
m = len(y)
for i in range(num_iterations):
h = sigmoid(X.dot(theta))
gradient = X.T.dot(h-y)/m
theta=alpha*gradient
return theta
theta = gradient_descent(theta,X,y,alpha = 0.01, num_iterations = 1000)
def predict(theta,X):
h = sigmoid(X.dot(theta))
y_pred = np.where(h>=0.5,1,0)
return y_pred
y_pred = predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)
print('Accuracy:',accuracy)
print(y_pred)
print(y)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```


## Output:
![Screenshot 2024-09-20 141436](https://github.com/user-attachments/assets/f6f5680e-f220-4529-9ffd-bdfc23d1a82a)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
