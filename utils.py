import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans

IMAGE_WIDTH = 1000
# Initialize weights for selected features
INITIALIZE_WEIGHTS = {
    'rent': 5.0,
    'area': 4.0,
    'city': 1.0,
    'distance': 1.0,
    'floor_score': 1.0
    # 'max_floor': 0.05,
    # 'deposit': 0.05,
    # 'fridge': 0.05,
    # 'washing_machine': 0.05,
    # 'air_conditioner': 0.05,
    # 'bed': 0.05,
    # 'water_heater': 0.05,
    # 'gas': 0.05,
    # 'closet': 0.05,
    # 'internet': 0.05,
    # 'sofa': 0.05,
    # 'table': 0.05,
    # 'tv': 0.05,
    # 'cable': 0.05,
    # 'balcony': 0.05,
    # 'park': 0.05,
    # 'electric_bill': 0.05,  # Negative weight for higher bills
    # 'water_bill': 0.05,      # Negative weight for higher bills
    # 'other_bill': 0.05,        # Negative weight for higher bills
    # 'garbage_bill': 0.05,     # Negative weight for higher bills
}

def get_floor_score(floor, max_floor):
    """Calculate a score based on the apartment's floor level."""
    if floor == -1 or floor > max_floor:
        return -0.1 * INITIALIZE_WEIGHTS['floor_score']
    else:
        return (1 - floor) * INITIALIZE_WEIGHTS['floor_score']
    
# Function to calculate scores based on weights
def calculate_scores(data, weights):
    """Calculate scores for each apartment based on given weights."""
    # Normalize and score features
    numeric_features = ['rent', 'area', 'floor', 'max_floor', 'deposit', 'distance',
                        'electric_bill', 'water_bill', 'other_bill', 'garbage_bill', 
                        'price_per_area', 'floor_score', 'city']
    scaler = MinMaxScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    
    # Compute total score
    data['score'] = (
        (1 - data['rent']) * weights['rent'] +
        data['area'] * weights['area'] +    # Area is rewarded
        (1 - data['city']) * weights['city'] +
        (1 - data['distance']) * weights['distance'] +  # Distance is penalized
        data['floor_score'] * weights['floor_score']

        # data['air_conditioner'] * weights['air_conditioner'] +
        # data['fridge'] * weights['fridge'] +
        # data['washing_machine'] * weights['washing_machine'] +
        # data['bed'] * weights['bed'] +
        # data['water_heater'] * weights['water_heater'] +
        # data['gas'] * weights['gas'] +
        # data['closet'] * weights['closet'] +
        # data['internet'] * weights['internet'] +
        # data['sofa'] * weights['sofa'] +
        # data['table'] * weights['table'] +
        # data['tv'] * weights['tv'] +
        # data['cable'] * weights['cable'] +
        # data['balcony'] * weights['balcony'] +
        # data['park'] * weights['park'] +
        # (1 - data['electric_bill']) * weights['electric_bill'] +  # Bills penalized
        # (1 - data['water_bill']) * weights['water_bill'] +
        # (1 - data['other_bill']) * weights['other_bill'] +
        # (1 - data['garbage_bill']) * weights['garbage_bill'] + 
    )
    return data

def train_model(scored_data, cv_folds=5):
    # Prepare data for the model after scores are calculated
    features = scored_data[['rent', 'area', 'city', 'distance', 'floor_score']]
    # features = scored_data.loc[:, ~scored_data.columns.isin(['item', 'link', 'score'])]
    target = scored_data['score']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Perform cross-validation
    cv_results = cross_validate(
        model, 
        X_train, 
        y_train, 
        cv=cv_folds, 
        scoring=['neg_mean_squared_error', 'r2'], 
        return_train_score=True
    )

    # Train the model on the full training set
    model.fit(X_train, y_train)
    # Predict on the testing data
    y_pred = model.predict(X_test)

    return model, cv_results, X_test, y_test, y_pred, features

def display_cv_metrics(cv_results):
    mse_scores = -cv_results['test_neg_mean_squared_error']
    r2_scores = cv_results['test_r2']

    st.subheader("Cross-Validated Performance Metrics")
    st.write(f"**Mean Squared Error (MSE):** {mse_scores.mean():.4f} ± {mse_scores.std():.4f}")
    st.write(f"**R² Score:** {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(data=mse_scores, ax=ax[0])
    ax[0].set_title("Cross-Validated MSE")
    ax[0].set_xlabel("MSE")

    sns.boxplot(data=r2_scores, ax=ax[1])
    ax[1].set_title("Cross-Validated R²")
    ax[1].set_xlabel("R²")

    plt.tight_layout()
    # Save the figure to a file
    plt.savefig("cv_metrics.png", bbox_inches='tight')
    plt.close()

    # Display the saved image in Streamlit with specific width
    st.image("cv_metrics.png", width=IMAGE_WIDTH)  # Adjust width as needed

def display_metrics(y_test, y_pred):
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.header(f'Mean Squared Error: {mse:.4f}')
    st.header(f'R2 Score: {r2:.4f}')

def plot_feature_importances(model, features):
    # Extract feature importances and create a DataFrame
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': features.columns,
        'importance': feature_importances
    })

    # Sort the DataFrame by importance in descending order
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    st.subheader("Feature Importances")

    # Set figure size and save the figure
    plt.figure(figsize=(8, 4))  # Adjust these values as needed
    sns.barplot(x=importance_df['importance'], y=importance_df['feature'])
    plt.title("Feature Importances in Random Forest Model")
    plt.xlabel("Importance")
    plt.ylabel("Features")

    # Save the figure to a file
    plt.savefig("feature_importances.png", bbox_inches='tight')
    plt.close()

    # Display the saved image in Streamlit with specific width
    st.image("feature_importances.png", width=IMAGE_WIDTH)  # Adjust width as needed

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def perform_clustering(data, n_clusters=3):
    features = data[
        [
            'rent',
            'area',
            'floor_score', 
            'distance',
            'city'
            ]
        ]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(features)
    return data, kmeans

# def plot_clusters(data):
#     # Display clusters on a scatter plot
#     st.subheader("Apartment Clusters")
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(
#         data=data, 
#         x='area', 
#         y='rent', 
#         hue='city', 
#         size='score', 
#         sizes=(20, 200),
#         style='cluster',
#         palette='tab10'
#     )
#     plt.title("Apartment Clusters")
#     # Save the figure to a file
#     plt.savefig("clusters.png", bbox_inches='tight')
#     plt.close()

#     # Display the saved image in Streamlit with specific width
#     st.image("clusters.png", width=IMAGE_WIDTH)  # Adjust width as needed

def plot_clusters(data):
    # Ensure data contains a 'link' column for displaying URLs on hover
    if 'link' not in data.columns:
        st.write("The dataset doesn't contain a 'link' column.")
        return

    # Add a minimum size threshold to avoid near-zero sizes
    min_marker_size = 5
    data['plot_size'] = data['score'].apply(lambda x: max(x, 0.01)) * 100  # Scaling for visibility
    data['plot_size'] = data['plot_size'].clip(lower=min_marker_size)  # Ensure a minimum size

     # Create a new column with HTML-formatted clickable links
    data['clickable_link'] = data['link'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')

    # Display clusters on an interactive scatter plot with hover functionality
    # st.subheader("Apartment Clusters")
    fig = px.scatter(
        data, 
        x='area', 
        y='rent', 
        color='city', 
        size='cluster',
        size_max=10,
        hover_data={'clickable_link': True, 'score': True, 'area': True, 'rent': True},
        title="Apartment Clusters",
        labels={'area': 'Area (sq ft)', 'rent': 'Rent ($)'}
    )

    fig.update_traces(marker=dict(sizemin=4))
    fig.update_layout(width=IMAGE_WIDTH, height=600)
    
    # Display Plotly chart in Streamlit
    st.plotly_chart(fig)

def get_shap_explainer(model, X):
    """Create a SHAP explainer for the model."""
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X, check_additivity=False)
    return shap_values

def plot_shap_summary(shap_values, features):
    """Plot SHAP summary plot."""
    st.subheader("SHAP Summary Plot")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_bar.png", bbox_inches='tight')
    plt.close()

    # Display the saved image in Streamlit with specific width
    st.image("shap_bar.png", width=IMAGE_WIDTH)  # Adjust width as needed
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features, show=False)
    plt.tight_layout()
    # Save the figure to a file
    plt.savefig("shap_summary.png", bbox_inches='tight')
    plt.close()

    # Display the saved image in Streamlit with specific width
    st.image("shap_summary.png", width=IMAGE_WIDTH)  # Adjust width as needed

def generate_explanation(apartment, shap_values, features):
    """
    Generate a detailed explanation for an apartment based on SHAP values.
    
    Parameters:
    - apartment (Series): The apartment data.
    - shap_values (shap.Explanation): SHAP values for the apartment.
    - features (Series): Feature values for the apartment.
    
    Returns:
    - explanation (str): A formatted explanation string.
    """
    explanation = f"### Explanation for {apartment['link']}\n\n"
    explanation += f"**Overall Score:** {apartment['score']:.4f}\n\n"
    explanation += "**Feature Contributions:**\n\n"
    
    # Get top contributing features
    shap_df = pd.DataFrame({
        'Feature': features.index,
        'SHAP Value': shap_values.values
    }).sort_values(by='SHAP Value', key=lambda x: x.abs(), ascending=False)
    
    for _, row in shap_df.iterrows():
        feature = row['Feature']
        shap_val = row['SHAP Value']
        feature_val = features[feature]
        direction = "increases" if shap_val > 0 else "decreases"
        explanation += f"- **{feature}**: {direction} the score by {abs(shap_val):.4f} (Value: {feature_val})\n"
    
    return explanation
