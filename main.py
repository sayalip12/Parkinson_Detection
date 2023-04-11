import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

dataset=pd.read_csv('C:/Users/shree/Desktop/Project Final yr/Final dataset.csv')

print(dataset.head())

print(dataset.info())

#correlation data
f,ax=plt.subplots(figsize=(14,14))
sns.heatmap(dataset.corr(),annot=True,ax=ax,fmt=".2f")
plt.xticks(rotation=90)
plt.show()

dataset.hist(bins=50, figsize=(28,28))
plt.show()

y =  dataset.status.values

X = dataset.drop(['Name', 'status'], axis=1)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)


knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print('Test set prediction: \n {}'.format(y_pred))


knn.score(X_test, y_test) * 100
print(knn.score(X_test, y_test)*100)
print(X_test)
print(y_test)

user_input=input("Enter all 13 values separated by coma")
user_input=list(map(float, user_input.split(',')))
y_pred = knn.predict([user_input])
print(y_pred)

if y_pred==1:
    print("Person is affected with Parkinson")
else:
    print("Person is not affected with parkinson")



