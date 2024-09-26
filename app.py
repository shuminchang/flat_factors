import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import utils

# Set page configuration
st.set_page_config(layout="wide")

# Load the apartment data
apartments_data = pd.read_csv('data/apartment_choice.csv')

original_apartments_data = apartments_data.copy()

# Feature Engineering
apartments_data['price_per_square_meter'] = apartments_data['rent'] / apartments_data['area']

# Streamlit App
st.title("Apartment Scoring Web App")

# Sidebar with weight adjustment sliders
st.sidebar.header("Adjust Weights")
with st.sidebar:
    for feature in utils.INITIALIZE_WEIGHTS.keys():
        initial_value = utils.INITIALIZE_WEIGHTS[feature]
        utils.INITIALIZE_WEIGHTS[feature] = st.slider(f"Weight for {feature}", 0.0, 1.0, initial_value, 0.01)

# Calculate scores with adjusted weights
scored_apartments = utils.calculate_scores(apartments_data.copy(), utils.INITIALIZE_WEIGHTS)

# Sort apartments by score in descending order
ranked_apartments = scored_apartments.sort_values(by='score', ascending=False)

ranked_apartments['item'] = original_apartments_data['item']
ranked_apartments['rent'] = original_apartments_data['rent']
ranked_apartments['area'] = original_apartments_data['area']
ranked_apartments['distance'] = original_apartments_data['distance']
ranked_apartments['electric_bill'] = original_apartments_data['electric_bill']
ranked_apartments['water_bill'] = original_apartments_data['water_bill']
ranked_apartments['other_bill'] = original_apartments_data['other_bill']
ranked_apartments['garbage_bill'] = original_apartments_data['garbage_bill']

# Create a link column (modify this to fit your actual link structure)
# ranked_apartments['link'] = ranked_apartments['item'].apply(lambda x: f"[Link](https://your-apartment-website.com/apartment/{x})")

# Display the ranked apartments with the clickable link column
st.header("Ranked Apartments")
# Convert DataFrame to markdown format for clickable links
st.markdown(ranked_apartments[['item', 'rent', 'area', 'distance', 'electric_bill', 'water_bill', 'other_bill', 'garbage_bill', 'city', 'score', 'link']].to_markdown(index=False))

model, X_test, y_test, y_pred, features = utils.train_model(scored_apartments)

utils.display_metrics(y_test, y_pred)

# Extract and plot feature importances
feature_importances = model.feature_importances_
st.subheader("Feature Importances")

# Set figure size and save the figure
plt.figure(figsize=(8, 4))  # Adjust these values as needed
sns.barplot(x=feature_importances, y=features.columns)
plt.title("Feature Importances in Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Features")

# Save the figure to a file
plt.savefig("feature_importances.png", bbox_inches='tight')
plt.close()

# Display the saved image in Streamlit with specific width
st.image("feature_importances.png", width=800)  # Adjust width as needed

@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(ranked_apartments)

st.download_button(
    label="Download ranked apartments as CSV",
    data=csv,
    file_name='ranked_apartments.csv',
    mime='text/csv',
)
