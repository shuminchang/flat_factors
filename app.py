import pandas as pd
import numpy as np
import streamlit as st
# Set page configuration
st.set_page_config(layout="wide")
import utils


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
        utils.INITIALIZE_WEIGHTS[feature] = st.slider(
            f"Weight for {feature}", 
            0.0, 
            1.0, 
            initial_value, 
            0.01
        )

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

tab1, tab2, tab3, tab4 = st.tabs(["Ranked Apartments", "Model Insights", "Clusters", "Model Explain"])

with tab1:
    # Display the ranked apartments with the clickable link column
    st.header("Ranked Apartments")
    # Convert DataFrame to markdown format for clickable links
    st.markdown(
        ranked_apartments[
            [
                'item', 'rent', 'area', 'distance', 'electric_bill', 'water_bill', 'other_bill', 'garbage_bill', 'city', 'score', 'link'
            ]
        ].to_markdown(index=False)
        )
    # Download button
    csv = utils.convert_df(ranked_apartments)
    st.download_button(
        label="Download ranked apartments as CSV",
        data=csv,
        file_name='ranked_apartments.csv',
        mime='text/csv',
    )

model, cv_results, X_test, y_test, y_pred, features = utils.train_model(scored_apartments)

with tab2:
    st.header("Model Insights")
    # Display cross-validation metrics
    utils.display_cv_metrics(cv_results)
    utils.display_metrics(y_test, y_pred)
    utils.plot_feature_importances(model, features)

# Add clustering option
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3, step=1)
ranked_apartments, kmeans = utils.perform_clustering(ranked_apartments, n_clusters)

with tab3:
    # Display clusters on a scatter plot
    st.header("Apartment Clusters")
    utils.plot_clusters(ranked_apartments)

with tab4:
    shap_values = utils.get_shap_explainer(model, X_test)
    utils.plot_shap_summary(shap_values, X_test)

    # Display explanations for top 3 apartments
    st.subheader("Explanations for Top Apartments")
    top_apartments = ranked_apartments.head(3).copy()
    top_apartments = top_apartments.reset_index()
    for index, apartment in top_apartments.iterrows():
        shap_value = shap_values[index]
        features = X_test.iloc[index]
        explanation = utils.generate_explanation(apartment, shap_value, features)
        st.markdown(explanation)