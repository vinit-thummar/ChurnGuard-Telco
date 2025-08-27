<div align="right">

[1]: https://github.com/vinit-thummar
[2]: https://www.linkedin.com/in/vinit-thummar/
[3]: https://twitter.com/vinit_thummar

[![github](https://raw.githubusercontent.com/vinit-thummar/ChurnGuard-Telco/main/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/vinit-thummar/ChurnGuard-Telco/main/icons/linkedin.svg)][2]
[![twitter](https://raw.githubusercontent.com/vinit-thummar/ChurnGuard-Telco/main/icons/twitter.svg)][3]

</div>

# <div align="center">ChurnGuard-Telco: Telecom Customer Churn Prediction</div>

![Intro](https://github.com/vinit-thummar/ChurnGuard-Telco/blob/main/output/customer_churn_intro.jpeg?raw=true)

## What is Customer Churn?

Customer churn occurs when customers discontinue their services with a company. In the telecom industry, users frequently switch providers due to competitive offerings. Predicting churn is critical because **retaining existing customers is far cheaper than acquiring new ones**.

By identifying high-risk customers in advance, companies can implement targeted retention strategies, improve customer loyalty, and minimize revenue loss.

## Objectives:

- Calculate the percentage of churned customers versus active customers.
- Analyze key features contributing to customer churn.
- Build and evaluate machine learning models to predict churn accurately.

## Dataset:

[Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

### Dataset Includes:

- Churn status – whether a customer left in the last month
- Services subscribed – phone, internet, online security, backup, tech support, streaming, etc.
- Account info – tenure, contract type, payment method, monthly & total charges
- Customer demographics – gender, senior citizen status, dependents, partners

## Implementation:

**Libraries:** sklearn, pandas, NumPy, Matplotlib, Seaborn

---

## Exploratory Data Analysis (EDA):

### 1. Churn Distribution:

> ![Churn distribution](https://github.com/vinit-thummar/ChurnGuard-Telco/blob/main/output/churn_distribution.png?raw=true)  
> 26.6% of customers switched providers.

### 2. Churn by Gender:

> ![Churn by gender](https://github.com/vinit-thummar/ChurnGuard-Telco/blob/main/output/churn_gender.png?raw=true)  
> Both genders exhibit similar churn patterns.

### 3. Customer Contract Distribution:

> ![Contract distribution](https://github.com/vinit-thummar/ChurnGuard-Telco/blob/main/output/contract_distribution.png?raw=true)  
> Month-to-month customers have the highest churn rate (~75%), compared to 13% for one-year contracts and 3% for two-year contracts.

### 4. Payment Methods:

> ![Payment Methods](https://github.com/vinit-thummar/ChurnGuard-Telco/blob/main/output/payment_methods.png?raw=true)  
> Customers using electronic checks are more likely to churn.

### 5. Internet Services:

> ![Internet Services](https://github.com/vinit-thummar/ChurnGuard-Telco/blob/main/output/internet_services.png?raw=true)  
> Fiber optic users have higher churn than DSL users.

### 6. Dependents:

> ![Dependents](https://github.com/vinit-thummar/ChurnGuard-Telco/blob/main/output/dependents.png?raw=true)  
> Customers without dependents churn more frequently.

### 7. Online Security:

> ![Online Security](https://github.com/vinit-thummar/ChurnGuard-Telco/blob/main/output/online_security.png?raw=true)  
> Lack of online security increases churn probability.

### 8. Senior Citizen Status:

> ![Senior Citizen](https://github.com/vinit-thummar/ChurnGuard-Telco/blob/main/output/senior_citizen.png?raw=true)  
> Senior citizens churn more often.

### 9. Paperless Billing:

> ![Paperless Billing](https://github.com/vinit-thummar/ChurnGuard-Telco/blob/main/output/paperless_billing.png?raw=true)  
> Customers using paperless billing show higher churn.

### 10. Tech Support:

> ![Tech Support](https://github.com/vinit-thummar/ChurnGuard-Telco/blob/main/output/tech_support.png?raw=true)  
> Lack of tech support correlates with higher churn.

### 11. Monthly Charges, Total Charges, and Tenure:

> ![Charges & Tenure](https://github.com/vinit-thummar/ChurnGuard-Telco/blob/main/output/charges_tenure.png?raw=true)  
> Higher monthly charges and new customers are more likely to churn.

---

## Machine Learning Models and Evaluation:

![ML Models](https://github.com/vinit-thummar/ChurnGuard-Telco/blob/main/output/model_evaluation.png?raw=true)

### Models Tested:

- Logistic Regression  
- K-Nearest Neighbors  
- Naive Bayes  
- Decision Tree  
- Random Forest  
- AdaBoost  
- Gradient Boosting  
- Voting Classifier

### Voting Classifier (Final Model):

```python
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf1 = GradientBoostingClassifier()
clf2 = LogisticRegression()
clf3 = AdaBoostClassifier()
voting_clf = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft')
voting_clf.fit(X_train, y_train)
predictions = voting_clf.predict(X_test)
print("Final Accuracy Score:", accuracy_score(y_test, predictions))
