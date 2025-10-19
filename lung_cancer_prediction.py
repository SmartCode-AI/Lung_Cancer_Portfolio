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

#   Support Vector Machine (Linear Kernel)

# we are using AGE and SMOKING for plotting
X_plot_features = X[['AGE', 'SMOKING']]

svm_plot = SVC(kernel='linear', C=1.0)
svm_plot.fit(X_plot_features, y)

X_train_plot, X_test_plot, y_train_plot, y_test_plot = train_test_split(
    X_plot_features, y, test_size=0.2, random_state=42
)
svm_plot.fit(X_train_plot, y_train_plot)
y_pred_plot = svm_plot.predict(X_test_plot)
from sklearn.metrics import accuracy_score
accuracy_plot = accuracy_score(y_test_plot, y_pred_plot)
print("SVM (AGE & SMOKING) Accuracy Score:",accuracy_plot * 100,"%")

x_min, x_max = X_plot_features['AGE'].min() - 1, X_plot_features['AGE'].max() + 1
y_min, y_max = X_plot_features['SMOKING'].min() - 1, X_plot_features['SMOKING'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = svm_plot.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_plot_features['AGE'], X_plot_features['SMOKING'], c=y, cmap=plt.cm.coolwarm, s=40, edgecolors='k')
plt.title("Lung Cancer Prediction (SVM on AGE & SMOKING)")
plt.xlabel("Age")
plt.ylabel("Smoking (1=No, 2=Yes)")
plt.show()

#  Logistic Regression

logis = LogisticRegression(solver='liblinear', C=10)
logis.fit(X_train, y_train)
y_pred_log = logis.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)
print("Logistic Regression Accuracy score:",acc_log * 100,"%")

#   Random Forest Classifier

forest = RandomForestClassifier(n_estimators=50, random_state=0)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)
acc_forest = accuracy_score(y_test, y_pred_forest)
print("Random Forest Accuracy score:",acc_forest * 100,"%")

# Plot
n_features = X.shape[1]
plt.figure(figsize=(10, 6))
plt.barh(range(n_features), forest.feature_importances_, align='center', color='red')
plt.yticks(np.arange(n_features), X.columns)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Lung Cancer Feature Importance (Random Forest)")
plt.show(k,)

#   Linear Regressor (Single Variable: AGE)

X_age = X[['AGE']]
X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(X_age, y, test_size=0.2, random_state=42)

l_regressor = LinearRegression()
l_regressor.fit(X_age_train, y_age_train)
y_age_pred = l_regressor.predict(X_age_test)

