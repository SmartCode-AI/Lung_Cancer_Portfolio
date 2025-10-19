import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, explained_variance_score
from sklearn import tree

#   data.info() "to check null values"
#   data.head() "to check the convertions 0/1"

# Loading the survey lung cancer.csv
data = pd.read_csv('\survey lung cancer.csv')
  
# Encode categorical features
label_encoder = LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])
    
# Separating features (X) and target (y)
# last column "LUNG_CANCER"(target)
X = data.drop('LUNG_CANCER', axis=1)  
y = data['LUNG_CANCER']       

#Split data into 80% Training and 20% Testing Sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

#   Naive Bayes GaussianNB

Gnb = GaussianNB()
Gnb.fit(X_train, y_train)

y_pred = Gnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(y_pred)
print("Accuracy:",accuracy * 100,"%")

#  Logistic Regression

logis = LogisticRegression(solver='liblinear', C=10)
logis.fit(X_train, y_train)
y_pred_log = logis.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)
print("Logistic Regression Accuracy score:",acc_log * 100,"%")

#   Linear Regressor (Single Variable: AGE)

X_age = X[['AGE']]
X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(X_age, y, test_size=0.2, random_state=42)

l_regressor = LinearRegression()
l_regressor.fit(X_age_train, y_age_train)
y_age_pred = l_regressor.predict(X_age_test)

