import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt

# Load the data from T11-2.DAT
data = np.loadtxt('T11-2.DAT')

# Extract predictors and target variable
X = data[:, 2:4]  # Columns 3 and 4 are predictors (indices 2 and 3)
y = data[:, 0]    # Column 1 is the response variable (index 0)

# Map labels from {1,2} to {-1,1}
y = np.where(y == 1, -1, 1).astype(float)

# Parameters for the Gaussian Kernel SVM
C = 1.0       # Regularization parameter
gamma = 1e-4  # Kernel coefficient for RBF (you can adjust this value)

# Define the Gaussian (RBF) kernel function
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

# Compute the Kernel Matrix
m, n = X.shape
K = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        K[i, j] = y[i] * y[j] * gaussian_kernel(X[i], X[j], gamma)

# Convert into cvxopt format
P = cvxopt_matrix(K)
q = cvxopt_matrix(-np.ones((m, 1)))

# Construct G and h for the inequality constraints
G_std = -np.eye(m)
h_std = np.zeros(m)

G_slack = np.eye(m)
h_slack = np.ones(m) * C

G = np.vstack((G_std, G_slack))
h = np.hstack((h_std, h_slack))

# Convert G and h into cvxopt format with correct data type
G = cvxopt_matrix(G)
h = cvxopt_matrix(h)

# Equality constraints
A = cvxopt_matrix(y.reshape(1, -1))
b_cvxopt = cvxopt_matrix(np.zeros(1))

# Ensure all matrices are of type 'd'
P = cvxopt_matrix(P, tc='d')
q = cvxopt_matrix(q, tc='d')
G = cvxopt_matrix(G, tc='d')
h = cvxopt_matrix(h, tc='d')
A = cvxopt_matrix(A, tc='d')
b_cvxopt = cvxopt_matrix(b_cvxopt, tc='d')

# Setting solver parameters
cvxopt_solvers.options['show_progress'] = False
cvxopt_solvers.options['abstol'] = 1e-7
cvxopt_solvers.options['reltol'] = 1e-7
cvxopt_solvers.options['feastol'] = 1e-7

# Run solver
solution = cvxopt_solvers.qp(P, q, G, h, A, b_cvxopt)
alphas = np.ravel(solution['x'])

# Identify support vectors
sv = alphas > 1e-4
ind = np.arange(len(alphas))[sv]
alphas_sv = alphas[sv]
sv_X = X[sv]
sv_y = y[sv]

# Compute b
b = 0
for n in range(len(alphas_sv)):
    b_n = sv_y[n]
    sum_term = 0
    for i in range(len(alphas_sv)):
        sum_term += alphas_sv[i] * sv_y[i] * gaussian_kernel(sv_X[i], sv_X[n], gamma)
    b_n -= sum_term
    b += b_n
b /= len(alphas_sv)

# Number of support vectors on the margin and not on the margin
on_margin = (alphas > 1e-4) & (alphas < C - 1e-4)
not_on_margin = (alphas >= C - 1e-4)

num_on_margin = np.sum(on_margin)
num_not_on_margin = np.sum(not_on_margin)

# Display results for Part c
print('Part c: SVM with Gaussian Kernel')
print('--------------------------------')
print('Alphas (non-zero) =\n', alphas_sv)
print('Number of support vectors =', len(alphas_sv))
print('Number of support vectors on the margin =', num_on_margin)
print('Number of support vectors NOT on the margin =', num_not_on_margin)
print('b =', b)

# Part d: Prediction for a new observation using Gaussian Kernel SVM

# New observation
z = np.array([117, 407])

# Compute the decision function
decision_value = 0
for i in range(len(alphas_sv)):
    decision_value += alphas_sv[i] * sv_y[i] * gaussian_kernel(sv_X[i], z, gamma)
decision_value += b

prediction = np.sign(decision_value)

# Display results for Part d
print('\nPart d: Prediction for new observation')
print('--------------------------------------')
print(f"Decision function value: {decision_value}")
print(f"Prediction for z = {z} is: {prediction}")

# Interpretation
if prediction == 1:
    print("The model predicts class 1 (original label 2).")
else:
    print("The model predicts class -1 (original label 1).")
