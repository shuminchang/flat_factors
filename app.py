import pandas as pd
import numpy as np
import streamlit as st
import utils

# Set page configuration
st.set_page_config(layout="wide")

def get_feature_engineered_data():
    # Load the apartment data
    apartments_data = pd.read_csv('data/apartment_choice.csv')
    # Feature Engineering
    apartments_data['price_per_area'] = apartments_data['rent'] / apartments_data['area']
    apartments_data['floor_score'] = apartments_data.apply(lambda row: utils.get_floor_score(row['floor'], row['max_floor']), axis=1)
    return apartments_data

# Update weights based on sidebar sliders
def adjust_weights():
    """Create sliders for weight adjustment in the sidebar."""
    st.sidebar.header("Adjust Weights")
    for feature in utils.INITIALIZE_WEIGHTS.keys():
        initial_value = utils.INITIALIZE_WEIGHTS[feature]
        utils.INITIALIZE_WEIGHTS[feature] = st.sidebar.slider(
            f"Weight for {feature}", 
            1.0, 
            10.0, 
            initial_value, 
            1.0
        )

def get_prepared_data(data):
    original_apartments_data = data.copy()

    # Calculate scores with adjusted weights
    scored_apartments = utils.calculate_scores(data.copy(), utils.INITIALIZE_WEIGHTS)

    # Sort apartments by score in descending order
    ranked_apartments = scored_apartments.sort_values(by='score', ascending=False)

    ranked_apartments['item'] = original_apartments_data['item']
    ranked_apartments['rent'] = original_apartments_data['rent']
    # ranked_apartments['rent_norm'] = scored_apartments['rent']
    ranked_apartments['area'] = original_apartments_data['area']
    ranked_apartments['distance'] = original_apartments_data['distance']
    ranked_apartments['electric_bill'] = original_apartments_data['electric_bill']
    ranked_apartments['floor'] = original_apartments_data['floor']
    ranked_apartments['price_per_area'] = original_apartments_data['price_per_area']
    ranked_apartments['max_floor'] = original_apartments_data['max_floor']
    ranked_apartments['city'] = original_apartments_data['city']

    apartments_for_display = ranked_apartments[
            [
                'item', 
                'rent',
                # 'rent_norm', 
                'area', 
                'city',
                'distance', 
                'floor', 
                'max_floor', 
                'score', 
                'link'
            ]
        ]
    return scored_apartments, ranked_apartments, apartments_for_display

def display_ranked_apartments(data, data_for_display):
    # Display the ranked apartments with the clickable link column
    st.header("Ranked Apartments")

    # Convert DataFrame to markdown format for clickable links
    st.markdown(
        data_for_display.to_markdown(index=False)
        )

    # Download button
    csv = utils.convert_df(data)
    st.download_button(
        label="Download ranked apartments as CSV",
        data=csv,
        file_name='ranked_apartments.csv',
        mime='text/csv',
    )

def display_unselected_features(data, data_for_display):
    st.header("Unselected Features")

    unselected_features = data.columns.difference(data_for_display.columns).tolist()

    for feature in unselected_features:
        st.write(f"- {feature}")

def display_model_insights(model, cv_results, y_test, y_pred, features):
    st.header("Model Insights")
    # Display cross-validation metrics
    utils.display_cv_metrics(cv_results)
    utils.display_metrics(y_test, y_pred)
    utils.plot_feature_importances(model, features)

def display_model_explanation(model, X_test, ranked_data):
    shap_values = utils.get_shap_explainer(model, X_test)
    utils.plot_shap_summary(shap_values, X_test)

    # Display explanations for top 3 apartments
    st.subheader("Explanations for Top Apartments")
    top_apartments = ranked_data.head(3).copy()
    top_apartments = top_apartments.reset_index()
    for index, apartment in top_apartments.iterrows():
        shap_value = shap_values[index]
        features = X_test.iloc[index]
        explanation = utils.generate_explanation(apartment, shap_value, features)
        st.markdown(explanation)

def display_app():
    apartments_data = get_feature_engineered_data()
    
    scored_apartments, ranked_apartments, apartments_for_display = get_prepared_data(apartments_data)
    
    model, cv_results, X_test, y_test, y_pred, features = utils.train_model(scored_apartments)

    # Add clustering option
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3, step=1)
    ranked_apartments, kmeans = utils.perform_clustering(ranked_apartments, n_clusters)


    # Create a link column (modify this to fit your actual link structure)
    # ranked_apartments['link'] = ranked_apartments['item'].apply(lambda x: f"[Link](https://your-apartment-website.com/apartment/{x})")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Ranked Apartments", "Unselected Features", "Clusters", "Model Insights", "Model Explain"]
    )

    with tab1:
        display_ranked_apartments(ranked_apartments, apartments_for_display)

    with tab2:
        display_unselected_features(ranked_apartments, apartments_for_display)

    with tab3:
        # Display clusters on a scatter plot
        st.header("Apartment Clusters")
        utils.plot_clusters(ranked_apartments)

    with tab4:
        display_model_insights(model, cv_results, y_test, y_pred, features)


    with tab5:
        display_model_explanation(model, X_test, ranked_apartments)

# Streamlit App
st.title("Apartment Scoring Web App")
adjust_weights()
display_app()