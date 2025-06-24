# Concrete Compressive Strength Prediction Project

## Project Overview
This project demonstrates a complete data science workflow, from data loading and cleaning to exploratory data analysis (EDA), visualization, and building a predictive machine learning model. The goal is to predict the compressive strength of concrete based on its components and age.

## Dataset
The dataset used is the Concrete Compressive Strength dataset, obtained from the UCI Machine Learning Repository. It contains 9 attributes: 8 input variables (components and age) and 1 output variable (compressive strength).

## Key Steps:
1.  **Data Loading:** Loaded the dataset from an Excel (.xls) file using Pandas.
2.  **Data Cleaning:**
    *   Identified and corrected inconsistent column names (e.g., extra spaces, trailing spaces).
    *   Handled missing values (though none were found in this dataset).
    *   Removed duplicate rows.
3.  **Exploratory Data Analysis (EDA):**
    *   Generated descriptive statistics (`.describe()`, `.info()`).
    *   Calculated custom summary statistics (mean, median, std, min, max) for all numerical features.
4.  **Data Visualization:**
    *   Histograms to understand the distribution of each feature.
    *   Scatter plots to visualize relationships between key components and concrete strength (e.g., Cement vs. Strength, Age vs. Strength).
    *   A correlation heatmap to identify the strength and direction of linear relationships between all numerical features.
5.  **Machine Learning Model:**
    *   Implemented a **Linear Regression** model to predict `Compressive_Strength_MPa`.
    *   Split the data into training (80%) and testing (20%) sets.
    *   Evaluated model performance using Mean Absolute Error (MAE) and R-squared (R²) score.
    *   Visualized actual vs. predicted strength to assess model accuracy.
6.  **Feature Importance:** Examined the coefficients of the Linear Regression model to understand the impact of each ingredient on concrete strength.

## How to Run the Project:
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-github-username/concrete-strength-ml-project.git
    cd concrete-strength-ml-project
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Ensure data file is present:** Download `Concrete_Data.xls` from the UCI Machine Learning Repository and place it in the project directory. **(Alternatively, you can include the .xls file in your repo if its size allows, like we did in these instructions).**
4.  **Run the script:**
    ```bash
    python concrete_analysis_project.py
    ```

## Results & Insights:
*(Briefly summarize some key findings here, e.g., "Cement and Age show strong positive correlations with compressive strength. The Linear Regression model achieved an R² of X.XX, indicating a reasonable fit...")*

## Optional Enhancements (Future Work):
*   Experiment with more advanced regression models (e.g., Decision Tree, Random Forest, Gradient Boosting) for potentially better performance.
*   Perform hyperparameter tuning for the chosen models.
*   Conduct more detailed feature engineering.
