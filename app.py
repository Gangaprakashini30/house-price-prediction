import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Streamlit App Title
st.title("House Price Analysis App")

# File Upload
st.sidebar.header("Upload Your Excel File")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file:
    # Load the dataset
    df = pd.read_excel(uploaded_file)

    # Data Collection
    st.header("Data Overview")
    st.write("## Initial Data")
    st.write(df.head())

    # Data Preprocessing
    st.header("Data Preprocessing")
    
    # Drop Duplicates
    st.write("Removing duplicates...")
    df.drop_duplicates(inplace=True)

    # Handle Outliers in 'SqFt' using IQR
    Q1 = df['SqFt'].quantile(0.25)
    Q3 = df['SqFt'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df['SqFt'] = df['SqFt'].clip(lower, upper)

    # Create 'Luxury_Index' feature
    df['Brick'] = df['Brick'].map({'Yes': 1, 'No': 0})
    df['Luxury_Index'] = (df['Bathrooms'] + df['Bedrooms']) * df['Brick']
    df['Price_per_sqft'] = df['Price'] / df['SqFt'].replace(0, np.nan)

    # Normalize numerical features
    scaler = MinMaxScaler()
    num_cols = ['SqFt', 'Bedrooms', 'Bathrooms', 'Price', 'Price_per_sqft']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Data Info and Description
    st.write("## Data Info")
    st.write(df.info())
    st.write("## Data Description")
    st.write(df.describe())

    # Data Visualization
    st.header("Data Visualization")

    # Plot Price Distribution
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Price'], bins=30, kde=True, color='teal', ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='Blues', linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Pairplot
    st.subheader("Pairplot of Key Features")
    sns.pairplot(df[num_cols])
    st.pyplot()

    # Boxplot: Bedrooms vs Price
    st.subheader("Price Distribution Across Bedroom Count")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Bedrooms', y='Price', data=df, palette='viridis', ax=ax)
    st.pyplot(fig)

    # Scatterplot: SqFt vs Price by Brick Status
    st.subheader("Price vs SqFt by Brick Status")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='SqFt', y='Price', hue='Brick', data=df, palette='viridis', ax=ax)
    st.pyplot(fig)

    # Display processed data
    st.write("## Processed Data")
    st.dataframe(df)

else:
    st.write("Please upload a valid Excel file to proceed.")
