X_plot_features = X[['AGE', 'SMOKING']]

svm_plot = SVC(kernel='linear', C=1.0)
svm_plot.fit(X_plot_features, y)

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

#   Logistic Regression

logis = LogisticRegression(solver='liblinear', C=10)
logis.fit(X_train, y_train)
y_pred_log = logis.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)
print("Logistic Regression Accuracy score:",acc_log * 100,"%")

#   Decision Tree Classifier

model = DecisionTreeClassifier(criterion="entropy", random_state=5)
model.fit(X_train, y_train)
y_pred_tree = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_tree)
print("Decision Tree Accuracy score:",accuracy * 100,"%")

# Plot Decision Tree
plt.figure(figsize=(20, 10))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=["NO", "YES"],
    filled=True,
    rounded=True
)
plt.title("Lung Cancer Decision Tree")
plt.show()