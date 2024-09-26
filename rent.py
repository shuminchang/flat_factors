import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Load the new apartment data
file_path = 'apartment_choice.csv'
apartments_data = pd.read_csv(file_path)

# Assign initial weights to all features
weights = {
    'rent': 0.4,
    'area': 0.3,
    'floor': 0.05,
    'max_floor': 0.05,
    'deposit': 0.05,
    'distance': 0.2,
    'fridge': 0.05,
    'washing_machine': 0.05,
    'air_conditioner': 0.05,
    'bed': 0.05,
    'water_heater': 0.05,
    'gas': 0.05,
    'closet': 0.05,
    'internet': 0.05,
    'sofa': 0.05,
    'table': 0.05,
    'tv': 0.05,
    'cable': 0.05,
    'balcony': 0.05,
    'park': 0.05,
    'electric_bill': 0.05,  # Negative weight for higher bills
    'water_bill': 0.05,      # Negative weight for higher bills
    'gas_bill': 0.05,        # Negative weight for higher bills
    'garbage_bill': 0.05     # Negative weight for higher bills
}

# Normalize relevant numeric features
numeric_features = ['rent', 'area', 'floor', 'max_floor', 'deposit', 'distance',
                    'electric_bill', 'water_bill', 'gas_bill', 'garbage_bill']
scaler = MinMaxScaler()
apartments_data[numeric_features] = scaler.fit_transform(apartments_data[numeric_features])

# Calculate score for each apartment based on weighted preferences
total_score = (
    (1 - apartments_data['rent']) * weights['rent'] +  # Rent is penalized
    apartments_data['area'] * weights['area'] +       # Area is rewarded
    (1 - apartments_data['distance']) * weights['distance'] +  # Distance is penalized
    apartments_data['air_conditioner'] * weights['air_conditioner'] +
    apartments_data['fridge'] * weights['fridge'] +
    apartments_data['washing_machine'] * weights['washing_machine'] +
    apartments_data['bed'] * weights['bed'] +
    apartments_data['water_heater'] * weights['water_heater'] +
    apartments_data['gas'] * weights['gas'] +
    apartments_data['closet'] * weights['closet'] +
    apartments_data['internet'] * weights['internet'] +
    apartments_data['sofa'] * weights['sofa'] +
    apartments_data['table'] * weights['table'] +
    apartments_data['tv'] * weights['tv'] +
    apartments_data['cable'] * weights['cable'] +
    apartments_data['balcony'] * weights['balcony'] +
    apartments_data['park'] * weights['park'] +
    (1 - apartments_data['electric_bill']) * weights['electric_bill'] +  # Bills penalized
    (1 - apartments_data['water_bill']) * weights['water_bill'] +
    (1 - apartments_data['gas_bill']) * weights['gas_bill'] +
    (1 - apartments_data['garbage_bill']) * weights['garbage_bill']
)

apartments_data['score'] = total_score

# Sort apartments by score in descending order
ranked_apartments = apartments_data.sort_values(by='score', ascending=False)

component_scores = pd.DataFrame({
    'rent_score': (1 - apartments_data['rent']) * weights['rent'],
    'area_score': apartments_data['area'] * weights['area'],
    'distance_score': (1 - apartments_data['distance']) * weights['distance'],
    'air_conditioner_score': apartments_data['air_conditioner'] * weights['air_conditioner'],
    'fridge_score': apartments_data['fridge'] * weights['fridge'],
    'washing_machine_score': apartments_data['washing_machine'] * weights['washing_machine'],
    'bed_score': apartments_data['bed'] * weights['bed'],
    'water_heater_score': apartments_data['water_heater'] * weights['water_heater'],
    'gas_score': apartments_data['gas'] * weights['gas'],
    'closet_score': apartments_data['closet'] * weights['closet'],
    'internet_score': apartments_data['internet'] * weights['internet'],
    'sofa_score': apartments_data['sofa'] * weights['sofa'],
    'table_score': apartments_data['table'] * weights['table'],
    'tv_score': apartments_data['tv'] * weights['tv'],
    'cable_score': apartments_data['cable'] * weights['cable'],
    'balcony_score': apartments_data['balcony'] * weights['balcony'],
    'park_score': apartments_data['park'] * weights['park'],
    'electric_score': (1 - apartments_data['electric_bill']) * weights['electric_bill'],
    'water_score': (1 - apartments_data['water_bill']) * weights['water_bill'],
    'gas_bill_score': (1 - apartments_data['gas_bill']) * weights['gas_bill'],
    'garbage_score': (1 - apartments_data['garbage_bill']) * weights['garbage_bill']
})

component_scores['score'] = total_score

component_scores = component_scores.sort_values(by='score', ascending=False)

# Combine with original data
ranked_apartments_with_scores = pd.concat([ranked_apartments, component_scores], axis=1)
# Show the ranked apartments with relevant features
ranked_apartments_with_scores.to_csv("ranked_apartments_with_scores.csv", index=False)

# Feature Engineering: Create a new feature 'price_per_square_meter'
apartments_data['price_per_square_meter'] = apartments_data['rent'] / apartments_data['area']

# Split the data into training and testing sets
features = apartments_data[numeric_features + ['fridge', 'washing_machine', 'air_conditioner',
                                                'bed', 'water_heater', 'gas', 'closet', 
                                                'internet', 'sofa', 'table', 'tv', 
                                                'cable', 'balcony', 'park']]
target = apartments_data['score']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
random_forest_model.fit(X_train, y_train)

# Predict on the testing data
y_pred = random_forest_model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Extract feature importances from the model
feature_importances = random_forest_model.feature_importances_
feature_names = X_train.columns
print(feature_importances)
print(feature_names)

# Create a bar plot for feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names, palette="viridis")
plt.title('Feature Importances in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()
