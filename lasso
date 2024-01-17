import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X = np.random.rand(100, 1) * 10  
y = 2 * X[:, 0] + 1 + np.random.randn(100)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alpha_values = [0.01, 0.1, 1.0, 10.0]

best_alpha = None
best_model = None
best_mse = float('inf')

for alpha in alpha_values:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)

    y_pred = lasso_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha
        best_model = lasso_model

print(f'Best Alpha: {best_alpha}')
print(f'Best MSE: {best_mse:.4f}')

plt.scatter(X, y, label='Data')
plt.plot(X, best_model.predict(X), color='red', label='Best-fit Line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Lasso Regression')
plt.legend()
plt.show()
