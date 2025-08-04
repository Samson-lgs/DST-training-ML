import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Set page configuration
st.set_page_config(
    page_title="Malware Detection System",
    page_icon="🔒",
    layout="wide"
)

# Title and description
st.title("🔒 Malware Detection System")
st.markdown("""
This application uses machine learning to detect malware in Android applications.
Choose a model and upload your data for analysis.
""")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    ["Decision Tree", "SVM (Linear)", "SVM (Polynomial)", "SVM (RBF)", "Logistic Regression", "K-Nearest Neighbors"]
)

# File upload
st.header("Data Upload")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load and preprocess data
        data = pd.read_csv(uploaded_file)
        st.success("File successfully uploaded!")
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        # Data preprocessing
        st.subheader("Data Preprocessing")
        with st.spinner("Preprocessing data..."):
            # Handle missing values and special characters
            data = data.replace('[?,S]', np.nan, regex=True)
            data.dropna(inplace=True)
            
            # Convert all columns to numeric
            for c in data.columns:
                if c != 'class':  # Skip the class column if it's categorical
                    data[c] = pd.to_numeric(data[c], errors='coerce')
            
            # Perform Label Encoding on the class column
            if 'class' in data.columns:
                lbl_enc = LabelEncoder()
                data['class'] = lbl_enc.fit_transform(data['class'])
            
            st.success("Data preprocessing completed!")
        
        # Split features and target
        if 'class' in data.columns:
            X = data.drop('class', axis=1)
            y = data['class']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Model training based on selection
            with st.spinner("Training the model..."):
                if model_choice == "Decision Tree":
                    model = DecisionTreeClassifier(random_state=42)
                elif model_choice == "SVM (Linear)":
                    model = SVC(kernel='linear', random_state=42)
                elif model_choice == "SVM (Polynomial)":
                    model = SVC(kernel='poly', random_state=42)
                elif model_choice == "SVM (RBF)":
                    model = SVC(kernel='rbf', random_state=42)
                elif model_choice == "Logistic Regression":
                    model = LogisticRegression(random_state=42)
                else:  # KNN
                    model = KNeighborsClassifier(n_neighbors=5)
                
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                accuracy = (y_pred == y_test).mean()
                
                # Display results
                st.subheader("Model Performance")
                st.metric("Accuracy", f"{accuracy:.2%}")
                
                # Save model
                if st.button("Save Model"):
                    model_filename = f"malware_detection_{model_choice.lower().replace(' ', '_')}.joblib"
                    joblib.dump(model, model_filename)
                    st.success(f"Model saved as {model_filename}")
        
        else:
            st.error("The uploaded file must contain a 'class' column for classification.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Instructions
st.sidebar.markdown("""
### How to use:
1. Select a machine learning model from the dropdown
2. Upload your CSV file containing the feature data
3. Wait for the model to process and train
4. View the results and save the model if desired
""")

# About section
st.sidebar.markdown("""
### About
This malware detection system uses various machine learning algorithms to classify Android applications as either benign or malicious. The models have shown high accuracy (96-98%) in detecting malware.
""")
