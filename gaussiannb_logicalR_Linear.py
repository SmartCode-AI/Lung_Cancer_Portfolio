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
