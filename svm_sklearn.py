from sklearn.svm import SVC

import numpy as np

# Load data
data = np.loadtxt('T11-2.DAT')

X = data[:, 2:4]
y = data[:, 0]

C = 1.0

# Fit SVM with linear kernel and specified C
clf = SVC(kernel='linear', C=C)
clf.fit(X, y.flatten())

# Get the number of support vectors
n_support_vectors = clf.n_support_.sum()
print('Number of support vectors (scikit-learn):', n_support_vectors)
