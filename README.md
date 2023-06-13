# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries and read the dataframe.
2.   Assign hours to X and scores to Y.
3.   Implement training set and test set of the dataframe.
4.   Plot the required graph both for test data and training data.
5.   Find the values of MSE , MAE and RMSE.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S.ANUSHARON
RegisterNumber:  212222240010


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')

df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

y_test
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scrores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scrores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```


## Output:

1.df.head()

![Screenshot (278)](https://github.com/Anusharonselva/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405600/7a6adddd-222b-459b-a131-20b32a154332)

2.df.tail()

![278 1](https://github.com/Anusharonselva/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405600/ec7005e5-434a-4111-aab0-42fc0a802636)

3.Array value of X

![Screenshot (280)](https://github.com/Anusharonselva/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405600/29373c6e-bed5-4e72-9e5f-bf2194658b50)

4.Array value of Y

![Screenshot (281)](https://github.com/Anusharonselva/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405600/69588fa7-63a0-4f51-9cb3-fd5dc7941ed7)

5.Values of Y prediction

![Screenshot (281) 1](https://github.com/Anusharonselva/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405600/87fa6b40-8743-4982-8a75-65cd8773f92b)

6.Array values of Y test

![Screenshot (281) 2](https://github.com/Anusharonselva/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405600/d633383d-8a1a-448b-b0bb-0a17e162df09)

7.Training Set Graph

![Screenshot (283)](https://github.com/Anusharonselva/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405600/932837c0-0a72-4a42-a604-fa92c300a55d)

8.Test Set Graph

![Screenshot (284)](https://github.com/Anusharonselva/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405600/83449558-5a22-434d-925f-f36466b762a8)

9.Values of MSE, MAE and RMSE

![Screenshot (285)](https://github.com/Anusharonselva/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405600/41533123-0f36-4884-889c-4e0741c7b85f)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
