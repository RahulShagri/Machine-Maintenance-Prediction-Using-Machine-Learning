import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from datetime import datetime

# Record start time
start_time = datetime.now()

# Load training data, extract the machine failed labels (last column), and delete the column from the original data
train_data = np.loadtxt("dataset\\ai4i2020_train_data.csv", delimiter=",")
train_data = shuffle(train_data, random_state=42)  # Randomize the data
train_data_labels = train_data[:, 6]
train_data = np.delete(train_data, 6, axis=1)

# Load testing data, extract the machine failed labels (last column), and delete the column from the original data
test_data = np.loadtxt("dataset\\ai4i2020_test_data.csv", delimiter=",")
test_data_labels = test_data[:, 6]
test_data = np.delete(test_data, 6, axis=1)

# Find number of total features
attributes = np.shape(train_data)[1]  # 6 Features

# Find covariance of the data
data_train_cov = np.cov(train_data.T)

# Calculate the eigen values and eigen vectors
eigenvalues, eigenvectors = np.linalg.eig(data_train_cov)

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

# Variables for storing errors
dtc_classification_error = []
rfc_classification_error = []

# Create instances of classifiers
clf_dtc = DecisionTreeClassifier()
clf_rfc = RandomForestClassifier()

# --- Decision Tree Classification and Random Forest Classification using PCA begins ---

# Iterate through dimensions to remove dimensions and calculate classification errors
for dimension in range(attributes, 0, -1):
    # Reduce the transformed data by removing dimensions
    transformed_data_reduced = transformed_train_data[:, 0:dimension]
    transformed_data_test_reduced = transformed_test_data[:, 0:dimension]

    # Decision Tree Classification and Random Forest Classification on the reduced data
    clf_dtc.fit(transformed_data_reduced, train_data_labels)  # Training of the model
    dtc_prediction = clf_dtc.predict(transformed_data_test_reduced)  # Testing of the model
    dtc_classification_error.append(sum(dtc_prediction != test_data_labels))  # Error Calculation

    clf_rfc.fit(transformed_data_reduced, train_data_labels)  # Training of the model
    rfc_prediction = clf_rfc.predict(transformed_data_test_reduced)  # Testing of the model
    rfc_classification_error.append(sum(rfc_prediction != test_data_labels))  # Error Calculation

print("\nDecision Tree Classification errors after PCA dimensionality reduction:\n")
for dimension in range(attributes, 0, -1):
    print(f"{dimension} retained dimensions: {dtc_classification_error[attributes - dimension]}")

print("\nRandom Forest Classification errors after PCA dimensionality reduction:\n")
for dimension in range(attributes, 0, -1):
    print(f"{dimension} retained dimensions: {rfc_classification_error[attributes - dimension]}")

# Plot the errors
x_axis = [i for i in range(6)]

pca_error_plot = plt.figure(1)
plt.plot(x_axis, dtc_classification_error)
plt.plot(x_axis, rfc_classification_error)
plt.xlabel("Retained dimensions out of 6 total dimensions in the dataset")
plt.ylabel("Number of errors")
plt.legend(["Decision Tree Classification Errors", "Random Forest Classification Errors"])
plt.xticks(ticks=[i for i in range(6)], labels=[i for i in range(6, 0, -1)])
plt.grid()
plt.title("Plot of Classification errors after dimensionality reduction using Principal Component Analysis")
pca_error_plot.show()

# Plot the eigenvalues or scores of PCA
x_axis_labels = [i for i in range(1, attributes+1)]

eigenvalues_plot = plt.figure(2)
plt.bar(x_axis_labels, np.flip(np.sort(eigenvalues)))
plt.xlabel("Principal Components")
plt.ylabel("Eigenvalues / PC Scores")
plt.title("Plot of Principal Component Scores")
plt.grid()
eigenvalues_plot.show()

# --- Decision Tree Classification and Random Forest Classification using PCA ends ---

# Record end time and show total run time
print(f"\nProgram completed in {(datetime.now() - start_time).seconds}."
      f"{(datetime.now() - start_time).microseconds} seconds.")

plt.show()
