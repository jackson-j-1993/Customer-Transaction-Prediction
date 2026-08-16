# Customer-Transaction-Prediction
A banking domain machine learning project aimed at predicting whether customers will make future transactions using 200 anonymized features. The project involves data preparation, handling class imbalance, and comparing classification models to recommend the best predictive model for production.

## Problem Statement

The objective of this project is to build a predictive model that helps the bank identify which customers are likely to make a transaction in the future. The dataset contains anonymized customer features, meaning the actual feature names are not available. The focus of the project lies mainly in data preparation, model building, model comparison, and providing a final recommendation for production.

## Project Tasks

1.  Prepare a complete data analysis report on the given data.
2.  Create a predictive model to identify customers who will make a transaction in the future.
3.  Compare multiple models and suggest the best model for production.
4.  Document the challenges faced and the techniques used to handle them.

## Dataset Overview

The dataset used is `train(1).csv` (or `train.csv`), consisting of 200,000 records and 202 columns. The structure is as follows:
*   `ID_code`: Unique customer ID (removed before training as it does not help in prediction).
*   `target`: The prediction column where 0 indicates the customer will not make a transaction, and 1 indicates the customer will make a transaction.
*   `var_0` to `var_199`: 200 anonymized numerical customer features.

### Data Quality & Class Imbalance
*   **Missing Values:** The dataset contains 0 missing values.
*   **Duplicates:** There are no duplicate records.
*   **Imbalance:** The data exhibits a heavy class imbalance, with 179,902 non-transaction customers (89.95%) and only 20,098 transaction customers (10.05%), resulting in an imbalance ratio of 8.95. 

## Tech Stack & Libraries

*   **pandas:** For reading and processing the dataset.
*   **numpy:** For numerical calculations.
*   **matplotlib & seaborn:** For creating charts and visualizations.
*   **scikit-learn (`sklearn`):** For building machine learning models, scaling data, stratifying train-test splits, and evaluating performance.

## Modeling & Evaluation

The data was split into training and testing sets (80-20 split) using stratified sampling to maintain the target class distribution. Because of the class imbalance, accuracy alone is misleading; therefore, the models were evaluated using Precision, Recall, F1 Score, and ROC-AUC. 

Three models were built and compared:
1.  **Logistic Regression:** Serves as a simple baseline model.
2.  **Random Forest Classifier:** Used to capture non-linear patterns.
3.  **Hist Gradient Boosting Classifier:** Powerful for tabular data and performs well on structured datasets.

### Best Model Selection
The **Logistic Regression** model was selected as the best model for production based primarily on its superior ROC-AUC score of 0.8599. ROC-AUC is especially useful here because it checks how well the model separates customers who will transact from those who will not.

## Challenges Faced & Techniques Used

*   **Challenge 1 - Anonymized Features:** The 200 features lacked business meaning.
    *   *Technique:* Focused on statistical and machine learning performance over business interpretation.
*   **Challenge 2 - Class Imbalance:** Non-transaction customers highly outnumbered transaction customers.
    *   *Technique:* Used stratified train-test splits and evaluated with ROC-AUC, precision, recall, and F1 score instead of relying solely on accuracy.
*   **Challenge 3 - High Number of Features:** The dataset contained 200 numerical features.
    *   *Technique:* Utilized models capable of handling high dimensionality and checked feature importance.
*   **Challenge 4 - ID Column:** `ID_code` was a non-predictive unique identifier.
    *   *Technique:* Dropped the column prior to training.
*   **Challenge 5 - Model Selection:** Different models yield varying results on the same data.
    *   *Technique:* Systematically trained and compared three different models under the same conditions.

## Production Recommendation

The selected Logistic Regression model helps identify high-probability customers. The bank can use this to:
*   Target high-probability customers for marketing campaigns.
*   Reduce unnecessary costs by skipping low-probability customers.
*   Prioritize engagement using data-driven probability scores.
*   Retrain the model regularly as new transaction data comes in.
