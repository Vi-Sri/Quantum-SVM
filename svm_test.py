import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt

# Load the data from T11-2.DAT
data = np.loadtxt('T11-2.DAT')

# Extract predictors and target variable
X = data[:, 2:4]  # Columns 3 and 4 are predictors
y = data[:, 0]    # Column 1 is the response variable

# Map labels from {1,2} to {-1,1}
y = np.where(y == 1, -1, 1)

# Visualize the data
plt.figure(figsize=(8,6))
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label='Class -1')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data Visualization')
plt.legend()
plt.savefig('svm_data_visualization.png')
plt.show()

# Initialize values and compute H
m, n = X.shape
y = y.reshape(-1,1) * 1.
X_dash = y * X
H = np.dot(X_dash , X_dash.T) * 1.

# Convert into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))

# Set regularization parameter C for soft-margin SVM
C = 1.0  # You can adjust C to see its effect on the model

# Construct G and h for the inequality constraints
G_std = -np.eye(m)
h_std = np.zeros(m)

G_slack = np.eye(m)
h_slack = np.ones(m) * C

G = cvxopt_matrix(np.vstack((G_std, G_slack)))
h = cvxopt_matrix(np.hstack((h_std, h_slack)))

# Equality constraints
A = cvxopt_matrix(y.reshape(1, -1))
b_cvxopt = cvxopt_matrix(np.zeros(1))

# Setting solver parameters (change default to decrease tolerance)
cvxopt_solvers.options['show_progress'] = False
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

# Run solver
solution = cvxopt_solvers.qp(P, q, G, h, A, b_cvxopt)
alphas = np.array(solution['x'])

# Compute the weight vector w
w = ((y * alphas).T @ X).reshape(-1,1)

# Identify support vectors
sv = (alphas > 1e-4).flatten()
print(f"Number of support vectors: {np.sum(sv)}")

# Compute the intercept b
b = y[sv] - np.dot(X[sv], w)
b = np.mean(b)

# Display results
print('Alphas = ', alphas[alphas > 1e-4].flatten())
print('w = ', w.flatten())
print('b = ', b)

# Plot the decision boundary
def plot_decision_boundary(X, y, w, b):
    plt.figure(figsize=(8,6))
    plt.scatter(X[y.flatten() == -1, 0], X[y.flatten() == -1, 1], color='red', label='Class -1')
    plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], color='blue', label='Class 1')
    plt.scatter(X[sv, 0], X[sv, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    # Create grid to evaluate model
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    XX, YY = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = (xy @ w + b).reshape(XX.shape)
    
    # Plot decision boundary and margins
    plt.contour(XX, YY, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary with Soft Margin')
    plt.legend()
    plt.savefig('svm_decision_boundary.png')
    plt.show()

# Call the plotting function
plot_decision_boundary(X, y, w, b)

# New observation
z = np.array([117, 407]).reshape(-1, 1)

# Print the new observation
print("-------------------------------")

# Compute the decision function
decision_value = np.dot(w.T, z) + b
prediction = np.sign(decision_value)
print(f"Prediction for z = {z.flatten()} is: {prediction[0][0]}")
print(f"Decision function value: {decision_value[0][0]}")


