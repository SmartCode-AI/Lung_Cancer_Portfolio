# Plot linear regression 
plt.scatter(X_age_test, y_age_test, color='red', label='Observed Lung Cancer')
plt.plot(X_age_test, y_age_pred, color='blue', linewidth=2, label='Predicted')
plt.xlabel('Age')
plt.ylabel('Lung Cancer Risk')
plt.title('Single Variable Linear Regression - Age vs Lung Cancer')
plt.legend()
plt.show()

# Linear performance
print("Performance of Linear Regressor:")
print("Mean absolute error =", round(mean_absolute_error(y_age_test, y_age_pred), 2))
print("Mean squared error =", round(mean_squared_error(y_age_test, y_age_pred), 2))
print("Median absolute error =", round(median_absolute_error(y_age_test, y_age_pred), 2))
print("Explained variance score =", round(explained_variance_score(y_age_test, y_age_pred), 2))
print("R2 score =", round(r2_score(y_age_test, y_age_pred), 2))
