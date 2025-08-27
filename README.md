<div align="right">
  
[1]: https://github.com/vinit-thummar
[2]: https://www.linkedin.com/in/vinit-thummar-58b731281/
[3]: https://public.tableau.com/
[4]: https://twitter.com/

[![github](https://raw.githubusercontent.com/vinit-thummar/ChurnGuard-Telco/main/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/vinit-thummar/ChurnGuard-Telco/main/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/vinit-thummar/ChurnGuard-Telco/main/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/vinit-thummar/ChurnGuard-Telco/main/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">ChurnGuard: Telecom Customer Churn Prediction</div>

![Intro](https://raw.githubusercontent.com/vinit-thummar/ChurnGuard-Telco/main/output/customer_churn_intro.png)

---

## What is Customer Churn?

Customer churn occurs when clients or subscribers stop using a company’s products or services. In telecom, churn is common due to high competition and multiple service provider options. Reducing churn is critical because retaining existing customers costs less than acquiring new ones.  

Predicting churn allows companies to focus on high-risk customers and apply retention strategies effectively, improving loyalty and revenue.  

---

## Objectives

- Calculate the percentage of churned vs active customers.  
- Analyze features that contribute most to churn.  
- Identify the most effective machine learning model for predicting churn.  

---

## Dataset

**[Telco Customer Churn Dataset](https://www.kaggle.com/bhartiprasad17/customer-churn-prediction/data)**

### Key Features

- Customer churn status (`Churn`)  
- Services subscribed: phone, internet, online backup, device protection, streaming, etc.  
- Account info: tenure, contract, payment method, billing, monthly/total charges  
- Demographics: gender, age range, partners, dependents  

---

## Implementation

**Libraries Used:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`  

**Steps:**

1. Data cleaning and preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Machine learning model training and evaluation  
4. Visualization of results and metrics  

---

## EDA Highlights

### 1. Churn Distribution
> ![Churn Distribution](https://raw.githubusercontent.com/vinit-thummar/ChurnGuard-Telco/main/output/churn_distribution.png)  
> Example: 26% of customers switched providers.  

### 2. Churn by Gender
> ![Churn by Gender](https://raw.githubusercontent.com/vinit-thummar/ChurnGuard-Telco/main/output/churn_by_gender.png)  
> Minimal difference between male and female churn rates.  

### 3. Contract Type
> ![Contract Distribution](https://raw.githubusercontent.com/vinit-thummar/ChurnGuard-Telco/main/output/contract_distribution.png)  
> Month-to-month contracts show higher churn compared to 1-year or 2-year contracts.  

### 4. Payment Method
> ![Payment Methods](https://raw.githubusercontent.com/vinit-thummar/ChurnGuard-Telco/main/output/payment_methods.png)  
> Customers paying via electronic check tend to churn more.  

### 5. Internet Services
> ![Internet Services](https://raw.githubusercontent.com/vinit-thummar/ChurnGuard-Telco/main/output/internet_services.png)  
> Fiber optic users show higher churn than DSL users.  

*(Add other plots such as dependent distribution, senior citizen churn, paperless billing, tech support, charges, and tenure.)*  

---

## Machine Learning Models

### Evaluated Models:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- AdaBoost  
- Voting Classifier (final model)  

### Voting Classifier Example:
```python
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

clf1 = GradientBoostingClassifier()
clf2 = LogisticRegression()
clf3 = AdaBoostClassifier()
voting_clf = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft')

voting_clf.fit(X_train, y_train)
predictions = voting_clf.predict(X_test)
print("Final Accuracy:", accuracy_score(y_test, predictions))
