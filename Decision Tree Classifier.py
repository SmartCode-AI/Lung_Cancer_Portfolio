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
