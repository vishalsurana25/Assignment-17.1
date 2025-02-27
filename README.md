**Project Title: Comparing Classifiers**

## **Jupyter Notebook : [Application Assignment 17.1](https://github.com/vishalsurana25/Assignment-17.1/blob/main/Model%20Comparison.ipynb)**

**1\. Introduction and Project Overview:**

This project leverages the UCI Bank Marketing Dataset (specifically, `bank-additional-full.csv`) to develop a predictive machine learning model. The primary objective is to accurately forecast whether a customer will subscribe to a long-term deposit product offered through telemarketing campaigns. This is a binary classification problem, where the target variable represents customer subscription (yes/no). We will employ a Jupyter notebook environment for data analysis, model development, and evaluation.

The project aims to achieve the following:

* **Develop a robust classification model:** Capable of predicting customer subscription to long-term deposits.  
* **Evaluate and compare multiple algorithms:** Specifically, K-Nearest Neighbors (KNN), Logistic Regression, Decision Trees, and Support Vector Machines (SVM).  
* **Analyze feature importance:** To understand which customer attributes significantly influence subscription decisions.  
* **Measure model performance:** Using metrics like accuracy and training time to identify the most effective model.

**2\. Problem Statement and Business Objectives:**

The dataset originates from a Portuguese banking institution's telemarketing campaigns conducted between May 2008 and November 2010\. Despite extensive efforts, the campaigns yielded a relatively low subscription rate of approximately 8% (6,499 successful subscriptions out of 79,354 contacts).

The core business objective is to enhance the success rate of future marketing campaigns by:

* Identifying key customer attributes that correlate with long-term deposit subscriptions.  
* Developing a predictive model to target potential subscribers more effectively.  
* Gaining insights into the factors influencing customer decisions, such as:  
  * The impact of existing loan products (housing, personal) on subscription rates.  
  * The relationship between educational level and subscription success.  
  * The effectiveness of different contact methods (cellular, telephone).

**3\. Dataset Description and Data Understanding:**

The dataset encompasses 17 marketing campaigns, resulting in 79,354 customer contacts. Each record includes detailed information about the customer, such as:

* **Demographic Information:** Job, marital status, education.  
* **Financial Information:** Default status, housing loan status, personal loan status.  
* **Target Variable:** Whether the customer subscribed to the long-term deposit ('yes' or 'no').

Initial analysis reveals a significant class imbalance, with a disproportionately low number of successful subscriptions.

**4\. Data Preparation and Preprocessing:**

To prepare the dataset for modeling, the following steps were performed:

* **Target Variable Renaming:** The original target variable "y" was renamed to "deposit" for clarity.  
* **Feature Selection:** A subset of relevant features (job, marital, education, default, housing, loan, contact) was selected for model training.  
* **Categorical and Numerical Feature Transformation:**  
  * `ColumnTransformer` was used to apply distinct preprocessing steps to categorical and numerical features.  
  * Categorical variables were one hot encoded.  
  * Numerical variables were scaled.  
* **Target Variable Encoding:** `LabelEncoder` was used to convert the categorical target variable (yes/no) into numerical values (0/1).  
* **Train-Test Split:** The dataset was divided into training and testing sets using `train_test_split`, with 30% of the data allocated for testing to evaluate model performance.

**5\. Model Development and Comparison:**

The following classification algorithms were implemented and compared:

* **K-Nearest Neighbors (KNN):** A non-parametric method that classifies data points based on their proximity to neighboring points.  
* **Logistic Regression:** A linear model that predicts the probability of a binary outcome.  
* **Decision Trees:** A tree-based model that makes predictions by recursively partitioning the data based on feature values.  
* **Support Vector Machines (SVM):** A powerful algorithm that finds the optimal hyperplane to separate data points into different classes.

For each model:

* The model was trained using the training dataset.  
* Predictions were made on the testing dataset.  
* Accuracy was calculated to evaluate model performance.  
* The training time was recorded.

**6\. Results and Findings:**

Based on the initial model comparison, Logistic Regression demonstrated superior performance, exhibiting the highest accuracy scores and the lowest training time. This suggests that Logistic Regression is the most efficient and accurate model for predicting long-term deposit subscriptions in this dataset, when using default parameters.

| Model | Train Time (s) | Train Accuracy | Test Accuracy |
| :---: | :---: | :---: | :---: |
| Logistic | 0.557 | 0.8872047448926502 | 0.8875940762320952 |
| KNN | 37.3 | 0.8858173493808748 | 0.8829813061422676 |
| Decision Tree | 0.702 | 0.8911935069890049 | 0.8846807477543093 |
| SVM | 49.8 | 0.8873087995560335 | 0.8875131504410455 |

**7\. Future Directions:**

* Address the class imbalance using techniques like oversampling or undersampling.  
* Perform hyperparameter tuning to optimize model performance.  
* Explore feature engineering to create new informative features.  
* Further analyze the feature importance of the best performing model.  
* Implement other metrics such as precision, recall, and F1 score.  
* Implement cross validation to further validate the results.
