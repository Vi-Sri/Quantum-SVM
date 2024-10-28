import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from qiskit import QuantumCircuit, Aer, execute
from qiskit.utils import QuantumInstance
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data from T11-2.DAT
data = np.loadtxt('T11-2.DAT')

# Extract predictors and target variable
X = data[:, 2:4]  # Columns 3 and 4 are predictors (indices 2 and 3)
y = data[:, 0]    # Column 1 is the response variable (index 0)

# Map labels from {1,2} to {-1,1}
y = np.where(y == 1, -1, 1).astype(float)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets (optional, you can use all data for training)
# For SVM with CVXOPT, we'll use all data due to the requirement of kernel matrix
X_train = X_scaled
y_train = y

# Parameters for the SVM
C = 1.0  # Regularization parameter

# Define the quantum feature map (data encoding circuit)
def quantum_feature_map(num_qubits):
    x = ParameterVector('x', num_qubits)
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
        qc.rz(x[i], i)
    return qc

# Initialize the quantum instance
backend = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend)

# Create the QuantumKernel object
num_features = X_train.shape[1]
feature_map = quantum_feature_map(num_features)
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

# Compute the kernel matrix using the quantum kernel
kernel_matrix = quantum_kernel.evaluate(x_vec=X_train)

# Set up the matrices for CVXOPT
m = y_train.shape[0]
Y = y_train.reshape(-1, 1)
H = np.outer(y_train, y_train) * kernel_matrix

P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G_std = -np.eye(m)
h_std = np.zeros(m)
G_slack = np.eye(m)
h_slack = np.ones(m) * C
G = cvxopt_matrix(np.vstack((G_std, G_slack)))
h = cvxopt_matrix(np.hstack((h_std, h_slack)))
A = cvxopt_matrix(y_train.reshape(1, -1))
b_cvxopt = cvxopt_matrix(np.zeros(1))

# Ensure matrices are of type 'd'
P = cvxopt_matrix(P, tc='d')
q = cvxopt_matrix(q, tc='d')
G = cvxopt_matrix(G, tc='d')
h = cvxopt_matrix(h, tc='d')
A = cvxopt_matrix(A, tc='d')
b_cvxopt = cvxopt_matrix(b_cvxopt, tc='d')

# Setting solver parameters
cvxopt_solvers.options['show_progress'] = True
cvxopt_solvers.options['abstol'] = 1e-5
cvxopt_solvers.options['reltol'] = 1e-5
cvxopt_solvers.options['feastol'] = 1e-5

# Run solver
solution = cvxopt_solvers.qp(P, q, G, h, A, b_cvxopt)
alphas = np.ravel(solution['x'])

# Support vectors have non zero lagrange multipliers
sv = alphas > 1e-4
ind = np.arange(len(alphas))[sv]
alphas_sv = alphas[sv]
sv_X = X_train[sv]
sv_y = y_train[sv]

print('Number of support vectors =', len(alphas_sv))

# Compute b
b = 0
for n in range(len(alphas_sv)):
    sum_term = 0
    for i in range(len(alphas_sv)):
        sum_term += alphas_sv[i] * sv_y[i] * kernel_matrix[ind[i], ind[n]]
    b_n = sv_y[n] - sum_term
    b += b_n
b /= len(alphas_sv)

print('Bias term b =', b)

# Function to make predictions
def predict(X_new):
    # Ensure X_new is scaled using the same scaler
    X_new_scaled = scaler.transform(X_new)
    K = quantum_kernel.evaluate(x_vec=X_new_scaled, y_vec=X_train)
    decision = np.dot((alphas * y_train), K.T) + b
    return np.sign(decision)

# Predict a new observation
z = np.array([[117, 407]])  # New data point (same as in previous parts)
prediction = predict(z)
print(f"Prediction for z = {z.flatten()} is: {prediction[0]}")

# Interpretation
if prediction[0] == 1:
    print("The model predicts class 1 (original label 2).")
else:
    print("The model predicts class -1 (original label 1).")
