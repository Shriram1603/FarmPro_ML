import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from sklearn.ensemble import GradientBoostingRegressor


data = pd.read_csv('Demand_forecast (1).csv')

print(data['State'].unique())

data['State']=data['State'].replace({'Tamil Nadu':9,'Assam':1,'Karnataka':2 ,'West Bengal':5 ,'punjab':16 ,'Kerala': 3,'Meghalaya':4})

print(data['Month'].unique())

data['Month']=data['Month'].replace({'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8,'September':9, 'October':10, 'November':11, 'December':12, 'january':1, 'february':2, 'march':3,'april':4, 'may':5, 'june':6, 'july':7, 'august':8, 'september':9, 'october':10, 'november':11,
 'december':12, 'Febuary':2, 'April ':4})

data['Crop']=data['Crop'].replace({'Maize':10,'Sugarcane':20,'Cotton':5 ,'Tobacco':23 ,'Paddy':17 ,'Barley':39 ,'Wheat':25,'Millets':19 ,'Oil seeds':52, 'Pulses':48 ,'Ground Nuts':32,'maize':10,'sugarcane':20,'cotton':5 ,'tobacco':23 ,'paddy':17 ,'barley':39 ,'wheat':25,'millets':19 ,'oil seed':52, 'pulses':48 ,'Ground nuts':32,'Groundnut':32,'Oilseeds':52,'Tabcco':23})

print(data['Crop'].unique())

# Split the data into training and testing sets
X = data[['State','Month','Crop']]
y = data['Demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

pickle.dump(model,open('demand.pkl','wb'))
model=pickle.load(open('demand.pkl','rb'))
