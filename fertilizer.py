import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

data = pd.read_csv('Fertilizer_Recommendation.csv')

print(data['Soil Type'].unique())

data['Soil Type']=data['Soil Type'].replace({'Sandy':1,'Loamy':2,'Black':3,'Red':4,'Clayey':5})
print(data['Soil Type'].unique())
print(data['Crop Type'].unique())
data['Crop Type']=data['Crop Type'].replace({'Maize':10,'Sugarcane':20,'Cotton':5 ,'Tobacco':23 ,'Paddy':17 ,'Barley':39 ,'Wheat':25,'Millets':19 ,'Oil seeds':52, 'Pulses':48 ,'Ground Nuts':32})

print(data['Fertilizer Name'].unique())

data['Fertilizer Name']=data['Fertilizer Name'].replace({'Urea': 1,'DAP':2 ,'14-35-14': 3,'28-28':4 ,'17-17-17': 5,'20-20': 6,'10-26-26':7})
print(data['Fertilizer Name'].unique())
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

#splitting

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)


model1=DecisionTreeClassifier(criterion='entropy',random_state=0)
model1.fit(x_train,y_train) 

y_pred=model1.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

pickle.dump(model1,open('fertilizer.pkl','wb'))
model=pickle.load(open('fertilizer.pkl','rb'))
