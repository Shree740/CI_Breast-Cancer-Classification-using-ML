import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the model and data
model = joblib.load('random_forest_model.pkl')  # Ensure the correct model file is used
data = pd.read_csv('breast_cancer_data.csv')  # Load breast cancer dataset

# Clean and prepare the dataset
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})  # Convert diagnosis to binary

# Model feature columns (adjusted to match dataset column names and training features)
feature_columns = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst',
    'fractal_dimension_worst'
]

# Streamlit app layout
st.title("Breast Cancer Prediction and Analysis")

# Sidebar for navigation
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Choose an option:", ["Home", "Prediction", "Analysis"])

if menu == "Home":
    st.subheader("Welcome to the Breast Cancer Prediction App")
    st.write("""
        This app uses a Random Forest model to predict whether a tumor is malignant or benign 
        based on user-provided input values. Explore the features of this app using the sidebar.
    """)

elif menu == "Prediction":
    st.subheader("Breast Cancer Diagnosis Prediction")
    user_input = {}

    # Input fields for user to provide tumor features
    for feature in feature_columns:
        user_input[feature] = st.number_input(
            f"{feature.replace('_', ' ').capitalize()}",
            min_value=0.0, max_value=2000.0, value=0.0, step=0.1
        )

    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    if st.button("Predict Diagnosis"):
        try:
            # Ensure input_df matches model feature names
            input_df = input_df[feature_columns]  # Align columns with those used in training
            diagnosis = model.predict(input_df)[0]
            diagnosis_label = "Malignant" if diagnosis == 1 else "Benign"

            # Show prediction result
            st.success(f"The predicted diagnosis is: **{diagnosis_label}**")
            
            # Show feature importance from the model
            st.subheader("Feature Importance")
            feature_importances = pd.DataFrame(
                model.feature_importances_, index=feature_columns, columns=['Importance']
            ).sort_values('Importance', ascending=False)
            st.bar_chart(feature_importances)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

elif menu == "Analysis":
    st.subheader("Dataset Analysis")
    st.write("Explore the key insights from the breast cancer dataset.")

    # Display dataset summary
    st.write("### Dataset Summary")
    st.dataframe(data.describe())

    # Show correlation heatmap
    st.write("### Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data[feature_columns + ['diagnosis']].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Diagnosis distribution
    st.write("### Diagnosis Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='diagnosis', palette='viridis', ax=ax)
    ax.set_xticklabels(['Benign', 'Malignant'])
    ax.set_title("Diagnosis Count")
    st.pyplot(fig)

    # Pairplot of key features
    st.write("### Pairplot of Features")
    selected_features = st.multiselect("Select features to include in pairplot:", feature_columns, default=feature_columns[:3])
    if selected_features:
        fig = sns.pairplot(data, vars=selected_features, hue='diagnosis', palette='viridis')
        st.pyplot(fig)

st.sidebar.write("For questions or feedback, contact the developer.")
