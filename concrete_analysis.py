# -----------------------------------------------------------------------------
# Project: Concrete Compressive Strength Prediction
# Author: Arham Ansari
# Date: [24-06-2025]
# Description: This script loads, cleans, analyzes, visualizes, and models
#              the Concrete Compressive Strength dataset.
# -----------------------------------------------------------------------------

# Part 0: Import Libraries
# These are the essential tools we'll use for data manipulation, visualization, and machine learning.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning specific imports
from sklearn.model_selection import train_test_split # To split data into training and testing sets
from sklearn.linear_model import LinearRegression    # Our chosen ML model
from sklearn.metrics import mean_absolute_error, r2_score # Metrics to evaluate model performance

# Set a style for Seaborn plots for better aesthetics
sns.set_style("whitegrid")

# -----------------------------------------------------------------------------
# Part 1: Configuration & File Loading
# We define the file path and then load the dataset.
# -----------------------------------------------------------------------------

# Define the full path to your .xls file using a raw string (r"...")
# This handles the backslashes correctly.
FILE_PATH = r"C:\Users\arham\Downloads\Project\Concrete_Data.xls"

def load_data(path):
    """
    Loads an Excel (.xls or .xlsx) file into a Pandas DataFrame.
    Handles potential FileNotFoundError and uses pd.read_excel().

    Args:
        path (str): The file path to the Excel dataset.

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    try:
        # Use pd.read_excel for .xls files. sheet_name=0 loads the first sheet.
        df = pd.read_excel(path, sheet_name=0)
        print(f"Successfully loaded data from: {path}")
        print(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        print("Please make sure the Excel file is in the correct directory.")
        print("You can download it from UCI Machine Learning Repository.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the data: {e}")
        print("Make sure you have 'openpyxl' and 'xlrd' installed: pip install openpyxl xlrd")
        return None

# Load the dataset
concrete_df = load_data(FILE_PATH)

# Exit if data loading failed
if concrete_df is None:
    exit()

# --- IMPORTANT: Print the original column names EXACTLY as loaded ---
print("\n--- ORIGINAL COLUMN NAMES (BEFORE ANY RENAMING ATTEMPT) ---")
print("Please COPY these names exactly if you need to update the column_names_map.")
original_columns_list = concrete_df.columns.tolist()
for col_name in original_columns_list:
    print(f"- '{col_name}'")
print("-" * 70) # Visual separator for clarity


# -----------------------------------------------------------------------------
# Part 2: Initial Data Exploration & Cleaning
# We'll inspect the data, identify issues, and clean it.
# -----------------------------------------------------------------------------

print("--- Initial Data Exploration ---")
print("\nFirst 5 rows of the dataset:")
print(concrete_df.head()) # Display the first 5 rows to get a quick look

print("\nDataset Information (Data Types, Non-Null Counts):")
concrete_df.info() # Get a summary of the DataFrame, including data types and non-null values

print("\nDescriptive Statistics (Numerical Columns):")
print(concrete_df.describe()) # Get summary statistics for numerical columns (mean, std, min, max, quartiles)

def clean_data(df):
    """
    Performs basic data cleaning steps:
    1. Renames columns for clarity (specific to this dataset, adjusted for .xls).
    2. Checks for and handles missing values (fills with median for numerical columns).
    3. Checks for and removes duplicate rows.

    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    cleaned_df = df.copy() # Always work on a copy to avoid modifying the original DataFrame directly

    print("\n--- Data Cleaning Process ---")
    print("\n--- Attempting Column Renaming ---")
    print("Columns BEFORE renaming attempt:", cleaned_df.columns.tolist())

    # 1. Rename columns for better readability (ADJUSTED FOR UCI .XLS NAMES based on your exact output)
    # The keys in this dictionary MUST EXACTLY match the original column names from your output.
    column_names_map = {
        'Cement (component 1)(kg in a m^3 mixture)': 'Cement_kg',
        'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'Blast_Furnace_Slag_kg',
        'Fly Ash (component 3)(kg in a m^3 mixture)': 'Fly_Ash_kg',
        'Water  (component 4)(kg in a m^3 mixture)': 'Water_kg', # Corrected: two spaces after Water
        'Superplasticizer (component 5)(kg in a m^3 mixture)': 'Superplasticizer_kg',
        'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'Coarse_Aggregate_kg', # Corrected: two spaces after Aggregate
        'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'Fine_Aggregate_kg',
        'Age (day)': 'Age_days',
        'Concrete compressive strength(MPa, megapascals) ': 'Compressive_Strength_MPa' # Corrected: trailing space
    }

    renamed_count = 0
    not_found_columns = []
    # Loop through the map to apply renaming
    for old_name, new_name in column_names_map.items():
        if old_name in cleaned_df.columns:
            cleaned_df.rename(columns={old_name: new_name}, inplace=True)
            print(f"  Renamed '{old_name}' to '{new_name}'")
            renamed_count += 1
        else:
            not_found_columns.append(old_name)
            print(f"  WARNING: Original column '{old_name}' (from map) NOT FOUND in DataFrame. Skipped renaming.")

    print("\nColumns AFTER renaming attempt:", cleaned_df.columns.tolist())

    if renamed_count > 0:
        print(f"\nSuccessfully renamed {renamed_count} columns.")
    if not_found_columns:
        print("\nThese columns from your 'column_names_map' were NOT found in the DataFrame (check for typos/spaces):")
        for col in not_found_columns:
            print(f"  - '{col}'")
        print("Please compare the above with the 'ORIGINAL COLUMN NAMES' list at the start of the script output.")


    # 2. Handle Missing Values
    print("\nChecking for Missing Values:")
    missing_values = cleaned_df.isnull().sum()
    print(missing_values[missing_values > 0]) # Print only columns with missing values

    if missing_values.sum() == 0:
        print("No missing values found. Great!")
    else:
        print("\nMissing values detected. Imputing numerical columns with their median...")
        # Using a loop to iterate through columns and fill missing values
        for col in cleaned_df.select_dtypes(include=np.number).columns:
            if cleaned_df[col].isnull().any():
                median_val = cleaned_df[col].median()
                cleaned_df[col].fillna(median_val, inplace=True)
                print(f"  Filled missing values in '{col}' with median: {median_val}")
        print("Missing values after imputation:")
        print(cleaned_df.isnull().sum()[cleaned_df.isnull().sum() > 0])


    # 3. Handle Duplicate Rows
    print("\nChecking for Duplicate Rows:")
    initial_rows = len(cleaned_df)
    duplicates_count = cleaned_df.duplicated().sum()
    print(f"Found {duplicates_count} duplicate rows.")

    if duplicates_count > 0:
        cleaned_df.drop_duplicates(inplace=True)
        print(f"Removed {duplicates_count} duplicate rows.")
        print(f"DataFrame now has {len(cleaned_df)} rows.")
    else:
        print("No duplicate rows found.")

    return cleaned_df

# Apply the cleaning function
concrete_df_cleaned = clean_data(concrete_df)

print("\n--- Data after Cleaning ---")
concrete_df_cleaned.info() # Verify data types and non-null counts again
print("\nDescriptive Statistics after cleaning:")
print(concrete_df_cleaned.describe()) # Check stats again (might change if imputation happened)


# -----------------------------------------------------------------------------
# Part 3: Data Analysis - Summary Statistics
# Let's get more specific summary statistics and store them.
# -----------------------------------------------------------------------------

def get_custom_summary_stats(df):
    """
    Calculates and stores custom summary statistics for numerical columns
    in a dictionary.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        dict: A dictionary where keys are column names and values are
              dictionaries of statistics (mean, median, std, min, max).
    """
    print("\n--- Custom Summary Statistics ---")
    summary_stats = {}
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Loop through each numerical column to calculate specific statistics
    for col in numerical_cols:
        col_stats = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std_dev': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
        summary_stats[col] = col_stats
        print(f"\n--- Statistics for '{col}' ---")
        # Loop through the dictionary items to print stats nicely
        for stat_name, stat_value in col_stats.items():
            print(f"  {stat_name.replace('_', ' ').title()}: {stat_value:.2f}")

    return summary_stats

# Get and print custom summary statistics
custom_stats = get_custom_summary_stats(concrete_df_cleaned)


# -----------------------------------------------------------------------------
# Part 4: Data Visualization
# Creating various plots to understand the data visually.
# -----------------------------------------------------------------------------

def plot_feature_distributions(df, columns):
    """
    Generates histograms for the distribution of specified numerical features.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list): A list of column names to plot.
    """
    print("\n--- Plotting Feature Distributions (Histograms) ---")
    num_plots = len(columns)
    # Determine grid size for subplots
    rows = int(np.ceil(num_plots / 3))
    cols = 3 if num_plots >= 3 else num_plots

    plt.figure(figsize=(cols * 5, rows * 4)) # Adjust figure size dynamically

    # Loop through each column to create a subplot
    for i, col in enumerate(columns):
        plt.subplot(rows, cols, i + 1)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

    plt.tight_layout() # Adjust subplot params for a tight layout
    plt.show()


def plot_relationships(df, x_col, y_col):
    """
    Generates a scatter plot to visualize the relationship between two numerical features.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The column name for the x-axis.
        y_col (str): The column name for the y-axis (target variable).
    """
    print(f"\n--- Plotting Relationship: {x_col} vs {y_col} (Scatter Plot) ---")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[x_col], y=df[y_col], alpha=0.6)
    plt.title(f'{y_col} vs. {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_correlation_heatmap(df):
    """
    Generates a heatmap of the correlation matrix for all numerical features.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
    """
    print("\n--- Plotting Correlation Heatmap ---")
    numerical_cols = df.select_dtypes(include=np.number).columns
    correlation_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Concrete Components and Strength')
    plt.show()


# Get list of numerical columns for plotting distributions
numerical_features = concrete_df_cleaned.select_dtypes(include=np.number).columns.tolist()

# Plot distributions of all numerical features
plot_feature_distributions(concrete_df_cleaned, numerical_features)

# Plot specific relationships against the target variable (Compressive_Strength_MPa)
target_variable = 'Compressive_Strength_MPa'

# Example relationships:
plot_relationships(concrete_df_cleaned, 'Age_days', target_variable)
plot_relationships(concrete_df_cleaned, 'Water_kg', target_variable)
plot_relationships(concrete_df_cleaned, 'Cement_kg', target_variable)

# Plot the correlation heatmap to see all relationships at once
plot_correlation_heatmap(concrete_df_cleaned)


# -----------------------------------------------------------------------------
# Part 5: Machine Learning Model - Linear Regression
# Training a simple regression model to predict concrete compressive strength.
# -----------------------------------------------------------------------------

print("\n--- Training Machine Learning Model ---")

# Define Features (X) and Target (y)
# X will be all columns except the 'Compressive_Strength_MPa' (our target)
X = concrete_df_cleaned.drop(columns=['Compressive_Strength_MPa'])
y = concrete_df_cleaned['Compressive_Strength_MPa']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# Split the data into training and testing sets
# test_size=0.2 means 20% of data will be for testing, 80% for training
# random_state for reproducibility (you get the same split every time)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Initialize and Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training complete.")

# Make Predictions on the Test Set
y_pred = model.predict(X_test)

# -----------------------------------------------------------------------------
# Part 6: Model Evaluation and Visualization
# Evaluating model performance and plotting actual vs. predicted values.
# -----------------------------------------------------------------------------

print("\n--- ML Model Evaluation ---")

# Mean Absolute Error (MAE): Average of the absolute differences between predictions and actual values.
# A lower MAE indicates a better fit.
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# R-squared (R² Score): Proportion of the variance in the dependent variable
# that is predictable from the independent variables. Closer to 1 is better.
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")

# Plot Actual vs Predicted Values
print("\n--- Plotting Actual vs Predicted Strength ---")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6) # Scatter plot of actual vs predicted
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--') # Red dashed line for ideal prediction (y=x)
plt.xlabel("Actual Compressive Strength (MPa)")
plt.ylabel("Predicted Compressive Strength (MPa)")
plt.title("Actual vs Predicted Concrete Strength (Linear Regression)")
plt.grid(True)
plt.show()

# Optional: Feature Importance (for Linear Regression, these are coefficients)
print("\n--- Model Coefficients (Feature Importance for Linear Regression) ---")
# Create a series to easily view coefficients with their corresponding feature names
coefficients = pd.Series(model.coef_, index=X.columns)
print(coefficients.sort_values(ascending=False))
print("Note: Positive coefficients indicate a positive relationship with strength, negative a negative relationship.")
print("The magnitude indicates the strength of the relationship.")


print("\n--- Project Analysis Complete ---")
print("This script covers data loading, cleaning, EDA, visualization, model training, and evaluation.")
