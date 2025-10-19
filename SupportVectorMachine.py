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
