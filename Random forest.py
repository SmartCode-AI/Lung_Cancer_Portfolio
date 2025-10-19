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
