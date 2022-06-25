"""After completing principal component analysis, it was found the the optimal number of dimensions without
significantly increasing the errors was 4."""
"""This code retains first 4 dimensions and performs Decision Tree and Random Forest Classification to calculate 
the errors in a confusion matrix."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Record start time
start_time = datetime.now()

# Load training data, extract the machine failed labels (last column), and delete the column from the original data
train_data = np.loadtxt("dataset\\ai4i2020_train_data.csv", delimiter=",")
train_data_labels = train_data[:, 6]
train_data = np.delete(train_data, 6, axis=1)

# Load testing data, extract the machine failed labels (last column), and delete the column from the original data
test_data = np.loadtxt("dataset\\ai4i2020_test_data.csv", delimiter=",")
test_data_labels = test_data[:, 6]
test_data = np.delete(test_data, 6, axis=1)

# Find number of total features
attributes = np.shape(train_data)[1]  # 6 Features

# Find covariance of the data
train_data_cov = np.cov(train_data.T)

# Calculate the eigen values and eigen vectors
eigenvalues, eigenvectors = np.linalg.eig(train_data_cov)

# Sort eigenvalues and store only indices
sorted_eigenvalue_indices = np.flip(np.argsort(eigenvalues))

# Sort eigenvectors using sorted eigenvalue indices
sorted_eigenvectors = np.empty(shape=(attributes, 0))
for dimension in range(attributes):
    sorted_eigenvectors = np.append(sorted_eigenvectors,
                                    eigenvectors[:, sorted_eigenvalue_indices[dimension]].reshape(attributes, 1),
                                    axis=1)

# Calculate transformed matrix of training using eigenvectors
transformed_train_data = np.dot(train_data, sorted_eigenvectors)

# Calculate transformed matrix of testing using eigenvectors
transformed_test_data = np.dot(test_data, sorted_eigenvectors)

# Variable to store confusion matrix errors
# Class A = Machine fails = 1
# Class B = Machine does NOT fail = 0
dtc_confusion_matrix = np.empty(shape=(2, 2))
dtc_confusion_matrix_percentage = np.empty(shape=(2, 2))
rfc_confusion_matrix = np.empty(shape=(2, 2))
rfc_confusion_matrix_percentage = np.empty(shape=(2, 2))

# Create instances of classifiers
clf_dtc = DecisionTreeClassifier()
clf_rfc = RandomForestClassifier()

# Reduce the transformed data by removing 2 dimensions and retaining 4
transformed_train_data_reduced = transformed_train_data[:, 0:4]
transformed_test_data_reduced = transformed_test_data[:, 0:4]

# Decision Tree Classification and Random Forest Classification on the reduced data
dtc_train_start_time = datetime.now()
clf_dtc.fit(transformed_train_data_reduced, train_data_labels)  # Training of the model
dtc_train_time = datetime.now() - dtc_train_start_time
dtc_test_start_time = datetime.now()
dtc_prediction = clf_dtc.predict(transformed_test_data_reduced)  # Testing of the model
dtc_test_time = datetime.now() - dtc_test_start_time

rfc_train_start_time = datetime.now()
clf_rfc.fit(transformed_train_data_reduced, train_data_labels)  # Training of the model
rfc_train_time = datetime.now() - rfc_train_start_time
rfc_test_start_time = datetime.now()
rfc_prediction = clf_rfc.predict(transformed_test_data_reduced)  # Testing of the model
rfc_test_time = datetime.now() - rfc_test_start_time

# Calculate confusion matrix values
for i in range (2):
    for j in range(2):
        dtc_confusion_matrix[i][j] = sum(np.logical_and(dtc_prediction == i, test_data_labels == j))
        rfc_confusion_matrix[i][j] = sum(np.logical_and(rfc_prediction == i, test_data_labels == j))

        if j == 0:
            dtc_confusion_matrix_percentage[i][j] = (dtc_confusion_matrix[i][j] / 2428)*100
            rfc_confusion_matrix_percentage[i][j] = (rfc_confusion_matrix[i][j] / 2428) * 100

        else:
            dtc_confusion_matrix_percentage[i][j] = (dtc_confusion_matrix[i][j] / 72) * 100
            rfc_confusion_matrix_percentage[i][j] = (rfc_confusion_matrix[i][j] / 72) * 100

# Print confusion matrices and accuracies
"""
Class B True -ve    Class B False +ve
Class A False -ve   Class A True +ve
"""

print("Decision Tree Classification Confusion Matrix:")
print(dtc_confusion_matrix)

print("\nDecision Tree Classification Confusion Matrix in percentages:")
print(dtc_confusion_matrix_percentage)

print("\nRandom Forest Classification Confusion Matrix:")
print(rfc_confusion_matrix)

print("\nRandom Forest Classification Confusion Matrix in percentages:")
print(rfc_confusion_matrix_percentage)

print(f"\nDecision Tree Classification Accuracy: "
      f"{((sum(dtc_prediction == test_data_labels))/len(test_data_labels))*100}%")
print(f"Random Forest Classification Accuracy: "
      f"{((sum(rfc_prediction == test_data_labels))/len(test_data_labels))*100}%")

# Record end time and show total run time and training and testing time
print(f"\nProgram completed in {(datetime.now() - start_time).seconds}."
      f"{(datetime.now() - start_time).microseconds} seconds.")

print(f"\nDecision Tree Classification training time: {dtc_train_time.seconds}.{dtc_train_time.microseconds} seconds")
print(f"Decision Tree Classification testing time: {dtc_test_time.seconds}.{dtc_test_time.microseconds} seconds")

print(f"\nRandom Forest Classification training time: {rfc_train_time.seconds}.{rfc_train_time.microseconds} seconds")
print(f"Random Forest Classification testing time: {rfc_test_time.seconds}.{rfc_test_time.microseconds} seconds")

