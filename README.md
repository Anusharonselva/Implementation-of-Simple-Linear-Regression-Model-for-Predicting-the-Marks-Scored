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
![1](https://user-images.githubusercontent.com/119405600/237008440-b442bffc-c425-434a-9462-924050a710be.png)


![2](https://user-images.githubusercontent.com/119405600/237008449-8684c536-eab9-470d-b44f-cdb6d229e1b2.png)


![3](https://user-images.githubusercontent.com/119405600/237008459-b74ab177-d7fb-4429-b68d-fdd85a418192.png)


![4](https://user-images.githubusercontent.com/119405600/237008479-a90f5a28-fd42-4e35-b5ca-d4e1466ecca8.png)


![5](https://user-images.githubusercontent.com/119405600/237008489-524abc65-d5ee-4f1e-b04d-eacf5a90c5c8.png)


![6](https://user-images.githubusercontent.com/119405600/237008498-077b8a31-7013-4e44-8918-934c72176cea.png)


![7](https://user-images.githubusercontent.com/119405600/237008519-4df8163c-3b09-4d70-afec-bc38655bd16b.png)


![8](https://user-images.githubusercontent.com/119405600/237008530-fd4f9bef-9ecf-4aec-9da3-189871f5bfc3.png)


![9](https://user-images.githubusercontent.com/119405600/237008541-bb4ba6d2-7379-4f92-872f-e2ff7096d1fb.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
