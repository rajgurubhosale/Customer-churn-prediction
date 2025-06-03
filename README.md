# 📊 Telecom Customer Churn Prediction

### Buisness Problem:
Telecom comapnies suffers from the Voluntary churn,which occurs when customers leave the service provider
for a competitor. Telecom companies spend hundreds of dollars
to acquire a new customer and when that customer leaves, the company not only loses the future revenue
from that customer but also the resources spend to acquire that customer. Churn Reduces profitability.
Study shows that increasing retention rate by 5 % increase profits by 25% to 95%😲.
To achieve this, it’s crucial to identify the customers who are most likely to leave (churn),
so effective retention strategies can be implemented.

### 🎯 Business Goal
* Develop a churn prediction model to flag customers most likely to churn.
* Perform EDA (Exploratory Data Analysis) to identify key behavioral and service-related churn factors.
* Optimize retention spending by focusing only on high-risk customers.

### Telecom companies use two main approaches to tackle this problem:

1)Untargeted Approach:
Focuses on improving the product and using mass advertising to increase brand loyalty.<br>
(mass advertising can be costly without targeting the right customers. It also lacks personalization, reducing its effectiveness in retaining customers.)

2)Targeted Approach:
Identifies customers likely to churn and offers interventions to keep them.


### Role of Predictive Modeling:
In the targeted approach, companies try to predict which customers are likely to churn and offer them special programs or incentives. However, if the churn predictions are inaccurate, companies may waste money on customers who would have stayed. There are various predictive modeling techniques to forecast churn, such as neural networks, logistic regression, and survival analysis. 
logistic regression is ideal for binary outcomes (churn/no churn), while neural networks can capture more complex relationships in large datasets.

### Evalution Metrics:
* **Primay Metric:** Recall- We want to catch as many churners possible as even if it means including some false positives.
* **Secondary Metrics:** Precision - useful for minimizing the cost of wrongly targeting non-churners.
* **Why Recall is Primary:** Missing a real churner means lost revenue, while wrongly flagging a loyal customer results in a smaller cost (e.g., a discount)**

### Buisness Impact

1) High Recall, Moderate Precision = Catch more true churners, increase retention ROI.
2) Supports data-driven decision-making for retention strategy.
3) Identifies likely churners early.

## RESULT

### Analysis Result

1) **Overall Churn Rate** is 0.146<br>
2) **The 415 area code** has a significantly larger customer base compared to 408 and 510 and still has same the churn rate as them ?. This may indicate stronger service quality, weaker competition, or a larger population in that region. Therefore, it would be strategic to invest more in customer retention and also targeted marketing efforts in the 415 area to maximize growth.<br>
3) **International Plan Pricing Issue:** There may be a data issue where customers who purchase the International Plan are being charged similarly to those without it, which could explain the higher churn rate. Customers may be dissatisfied if the plan does not offer the expected additional value, such as lower rates for international calls."<br>
4) **Customers with Vmail Plan:** they have very low churn rate **(0.088)** !!This could indicate that the service is highly valued, suggesting that enhancing or promoting this service further might improve customer retention.<br>
5) **Churn Increases After 3 Service Calls:** it increase drastically Customers with more calls might be experiencing unresolved issues, possibly due to insufficient attention from the company.
can significantly reduce churn if try to understand their problems and solve!...<br>

## Prediction Result
## 🔢 Model Performance
## 🤖 Algorithm Comparison

| Algorithm        | Recall | Precision |
|------------------|--------|-----------|
| Baseline(XGBoost)| 83%   | 75%        |
| Random Forest    | 87%   | 98%        |
| XGBoost          | 87%   | 98%        |


<details>
<summary><strong>Customer-churn-prediction/

├── app/                          # Streamlit application
│   ├── pages/
│   │   └── 1_batch_prediction.py
│   └── app.py                    # Main application file
│
├── data/                         # Data storage
│   ├── train.csv                 # Raw training data
│   ├── test.csv                  # Raw test data
│   ├── fe_train.csv              # Feature-engineered training data
│   ├── fe_test.csv               # Feature-engineered test data
│   └── test_sample.csv           # Sample of processed data
│
├── docs/                         # Documentation
│   └── project_overview.md       # Project overview documentation
│
├── models/                       # Machine learning models
│   ├── baseline_model.pkl
│   ├── rf_model.pkl              # Random Forest model
│   ├── gb_model.pkl              # Gradient Boosting model
│   └── final_model.pkl           # Final selected model
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   ├── 02_baseline_model.ipynb   # Baseline model development
│   ├── 03_feature_engineering.ipynb  # Feature engineering
│   ├── 04_model_experiment_evaluation.ipynb  # Model experiments and evaluation
│   └── 05_final_model.ipynb      # Final model training and evaluation
│
├── output/                       # Output files
│   ├── results_io.py             # Script to save/load results (?)
│   └── results_.csv              # Results data file
│
├── src/                          # Source code
│   ├── __init__.py               # Init file for src package
│   ├── feature_engineering.py    # Feature engineering functions
│   ├── metrics.py                # Evaluation metrics functions
│   ├── preprocessing.py          # Data preprocessing functions
│   └── training_model.py         # Model training functions
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project README
</strong></summary>




app link : https://customer-churn-prediction-apk.streamlit.app/
