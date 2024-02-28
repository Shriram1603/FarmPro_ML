import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
# Load the data
data = pd.read_csv('crop_yield.csv')

# Mapping crop names to numerical indices
crop_indices = {
    "Arecanut": 1,
    "Arhar/Tur": 2,
    "Castor seed": 3,
    "Coconut ": 4,
    "Cotton(lint)": 5,
    "Dry chillies": 6,
    "Gram": 7,
    "Jute": 8,
    "Linseed": 9,
    "Maize": 10,
    "Mesta": 11,
    "Niger seed": 12,
    "Onion": 13,
    "Other Rabi pulses": 14,
    "Potato": 15,
    "Rapeseed &Mustard": 16,
    "Rice": 17,
    "Sesamum": 18,
    "Small millets": 19,
    "Sugarcane": 20,
    "Sweet potato": 21,
    "Tapioca": 22,
    "Tobacco": 23,
    "Turmeric": 24,
    "Wheat": 25,
    "Bajra": 26,
    "Black pepper": 27,
    "Cardamom": 28,
    "Coriander": 29,
    "Garlic": 30,
    "Ginger": 31,
    "Groundnut": 32,
    "Horse-gram": 33,
    "Jowar": 34,
    "Ragi": 35,
    "Cashewnut": 36,
    "Banana": 37,
    "Soyabean": 38,
    "Barley": 39,
    "Khesari": 40,
    "Masoor": 41,
    "Moong(Green Gram)": 42,
    "Other Kharif pulses": 43,
    "Safflower": 44,
    "Sannhamp": 45,
    "Sunflower": 46,
    "Urad": 47,
    "Peas & beans (Pulses)": 48,
    "other oilseeds": 49,
    "Other Cereals": 50,
    "Cowpea(Lobia)": 51,
    "Oilseeds total": 52,
    "Guar seed": 53,
    "Other Summer Pulses": 54,
    "Moth": 55
}

# Replace crop names with numerical indices
data['Crop'] = data['Crop'].map(crop_indices)

# Remove leading and trailing spaces from 'Season' column
data['Season'] = data['Season'].str.strip()

# Replace 'Season' values with numerical indices
data['Season'] = data['Season'].replace({'Whole Year': 1, 'Kharif': 2, 'Rabi': 3, 'Autumn': 4, 'Summer': 5, 'Winter': 6})

# Mapping state names to numerical indices
state_indices = {
    'Assam': 1, 'Karnataka': 2, 'Kerala': 3, 'Meghalaya': 4, 'West Bengal': 5,
    'Puducherry': 6, 'Goa': 7, 'Andhra Pradesh': 8, 'Tamil Nadu': 9, 'Odisha': 10,
    'Bihar': 11, 'Gujarat': 12, 'Madhya Pradesh': 13, 'Maharashtra': 14, 'Mizoram': 15,
    'Punjab': 16, 'Uttar Pradesh': 17, 'Haryana': 18, 'Himachal Pradesh': 19, 'Tripura': 20,
    'Nagaland': 21, 'Chhattisgarh': 22, 'Uttarakhand': 23, 'Jharkhand': 24, 'Delhi': 25,
    'Manipur': 26, 'Jammu and Kashmir': 27, 'Telangana': 28, 'Arunachal Pradesh': 29, 'Sikkim': 30
}

# Replace state names with numerical indices
data['State'] = data['State'].replace(state_indices)

# Selecting features (X) and target variable (y)
X = data[['Crop', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
y = data['Yield']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



# Drop rows where NaN values occur in the "Crop" column
data = data.dropna(subset=['Crop'])

# Resetting the index after dropping rows
data.reset_index(drop=True, inplace=True)

# Check for NaN values in each column
nan_columns = data.columns[data.isna().any()].tolist()

# Print columns with NaN values
print("Columns with NaN values:", nan_columns)

# Initialize the SimpleImputer
imputer = SimpleImputer(strategy='mean')

# Fit the imputer to the training data and transform the data
X_train_imputed = imputer.fit_transform(X_train)

# Transform the test data
X_test_imputed = imputer.transform(X_test)

# Initialize and train the Gradient Boosting Regressor
model = GradientBoostingRegressor()
model.fit(X_train_imputed, y_train)

# Predict on the test set
y_pred = model.predict(X_test_imputed)



# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

pickle.dump(model,open('Yeild.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
