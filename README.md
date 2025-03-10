# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: G.Lahari sindhu
RegisterNumber: 212223240038 
*/
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```


## Output:
## dataset:
![image](https://github.com/user-attachments/assets/063188a3-0c83-4a3c-acab-c12e3431ec2c)
## Head Values:
![image](https://github.com/user-attachments/assets/4dbcc5e7-bba7-47d4-9b67-87c658bf9678)
## Tail Values:
![image](https://github.com/user-attachments/assets/937af52f-e68e-43b5-b654-502cfc2a94eb)
## X and Y values:
![image](https://github.com/user-attachments/assets/2df3e3a6-9945-41db-a6b4-a747a7f8fd7c)
## Predication values of X and Y:
![image](https://github.com/user-attachments/assets/0c512fc8-8054-4024-830f-64d95839a9a3)
## Training Set:
![image](https://github.com/user-attachments/assets/1790ca9f-a764-4c80-b36f-e90ca43adb2a)
![image](https://github.com/user-attachments/assets/f4da1385-1501-4e3a-b084-9c928da912e5)
## MSE,MAE and RMSE:
![image](https://github.com/user-attachments/assets/98b5303c-14c5-48f9-9f1b-945f50f25a5a)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
