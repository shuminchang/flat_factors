from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
# Initialize weights for selected features
INITIALIZE_WEIGHTS = {
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
    'other_bill': 0.05,        # Negative weight for higher bills
    'garbage_bill': 0.05     # Negative weight for higher bills
}

# Function to calculate scores based on weights
def calculate_scores(data, weights):
    # Normalize and score features
    # max_rent = data['rent'].max()
    # max_area = data['area'].max()
    # max_distance = data['distance'].max()
    
    # rent_score = ((max_rent - data['rent']) / max_rent) * weights['rent']
    # area_score = (data['area'] / max_area) * weights['area']
    # distance_score = ((max_distance - data['distance']) / max_distance) * weights['distance']
    numeric_features = ['rent', 'area', 'floor', 'max_floor', 'deposit', 'distance',
                    'electric_bill', 'water_bill', 'other_bill', 'garbage_bill']
    scaler = MinMaxScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    
    # Compute total score
    data['score'] = (
        (1 - data['rent']) * weights['rent'] +  # Rent is penalized
        data['area'] * weights['area'] +       # Area is rewarded
        (1 - data['distance']) * weights['distance'] +  # Distance is penalized
        data['air_conditioner'] * weights['air_conditioner'] +
        data['fridge'] * weights['fridge'] +
        data['washing_machine'] * weights['washing_machine'] +
        data['bed'] * weights['bed'] +
        data['water_heater'] * weights['water_heater'] +
        data['gas'] * weights['gas'] +
        data['closet'] * weights['closet'] +
        data['internet'] * weights['internet'] +
        data['sofa'] * weights['sofa'] +
        data['table'] * weights['table'] +
        data['tv'] * weights['tv'] +
        data['cable'] * weights['cable'] +
        data['balcony'] * weights['balcony'] +
        data['park'] * weights['park'] +
        (1 - data['electric_bill']) * weights['electric_bill'] +  # Bills penalized
        (1 - data['water_bill']) * weights['water_bill'] +
        (1 - data['other_bill']) * weights['other_bill'] +
        (1 - data['garbage_bill']) * weights['garbage_bill']
    )
    return data

def train_model(scored_data):
    # Prepare data for the model after scores are calculated
    # features = scored_apartments[['rent', 'area', 'distance', 'electric_bill', 'water_bill', 'other_bill', 'garbage_bill']]
    features = scored_data.loc[:, ~scored_data.columns.isin(['link', 'score'])]
    target = scored_data['score']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the testing data
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred, features

def display_metrics(y_test, y_pred):
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.header(f'Mean Squared Error: {mse:.4f}')
    st.header(f'R2 Score: {r2:.4f}')