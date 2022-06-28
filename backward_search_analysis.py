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

# Calculate number of features
attributes = np.shape(train_data)[1]  # 6

# Variables for storing errors
dtc_classification_error = []
rfc_classification_error = []

# Create instances of classifiers
clf_dtc = DecisionTreeClassifier()
clf_rfc = RandomForestClassifier()

# --- Decision Tree Classification and Random Forest Classification using backward search begins ---

# Running backward search for Decision Tree Classification
# Begin with all dimensions
train_data_reduced = train_data
test_data_reduced = test_data

print("\nDecision Tree Classification errors after backward search dimensionality reduction:\n")

# Iterate though all dimensions
for iteration in range(attributes):
    classification_error = []

    # Calculating error when all dimensions are retained
    if iteration == 0 or iteration == 5:
        # Decision Tree Classification and Random Forest Classification on the reduced data
        clf_dtc.fit(train_data_reduced, train_data_labels)  # Training of the model
        dtc_prediction = clf_dtc.predict(test_data_reduced)  # Testing of the model
        classification_error.append(sum(dtc_prediction != test_data_labels))  # Error Calculation

    # Calculating for rest of the dimensions
    else:
        for dimension in range(attributes - iteration - 1, -1, -1):
            train_data_dimension_reduced = np.delete(train_data_reduced, obj=dimension, axis=1)
            test_data_dimension_reduced = np.delete(test_data_reduced, obj=dimension, axis=1)

            # Decision Tree Classification and Random Forest Classification on the reduced data
            clf_dtc.fit(train_data_dimension_reduced, train_data_labels)  # Training of the model
            dtc_prediction = clf_dtc.predict(test_data_dimension_reduced)  # Testing of the model
            classification_error.append(sum(dtc_prediction != test_data_labels))  # Error Calculation

    # Find the minimum classification error
    dtc_classification_error.append(min(classification_error))
    print(f"{attributes - iteration} retained dimensions: {min(classification_error)}")

    # Find the index of the dimension of the least classification error
    dimension_to_remove = classification_error.index(min(classification_error))
    print(f"Index of removed dimension from the retained dimensions: {dimension_to_remove}")

    # Remove the dimension from the data
    train_data_reduced = np.delete(train_data_reduced, obj=dimension_to_remove, axis=1)
    test_data_reduced = np.delete(test_data_reduced, obj=dimension_to_remove, axis=1)

# Running backward search for Random Forest Classification
# Begin with all dimensions
train_data_reduced = train_data
test_data_reduced = test_data

print("\nRandom Forest Classification errors after backward search dimensionality reduction:\n")

# Iterate though all dimensions
for iteration in range(attributes):
    classification_error = []

    # Calculating error when all dimensions are retained
    if iteration == 0 or iteration == 5:
        # Decision Tree Classification and Random Forest Classification on the reduced data
        clf_rfc.fit(train_data_reduced, train_data_labels)  # Training of the model
        rfc_prediction = clf_rfc.predict(test_data_reduced)  # Testing of the model
        classification_error.append(sum(rfc_prediction != test_data_labels))  # Error Calculation

    # Calculating for rest of the dimensions
    else:
        for dimension in range(attributes - iteration - 1, -1, -1):
            train_data_dimension_reduced = np.delete(train_data_reduced, obj=dimension, axis=1)
            test_data_dimension_reduced = np.delete(test_data_reduced, obj=dimension, axis=1)

            # Decision Tree Classification and Random Forest Classification on the reduced data
            clf_rfc.fit(train_data_dimension_reduced, train_data_labels)  # Training of the model
            rfc_prediction = clf_rfc.predict(test_data_dimension_reduced)  # Testing of the model
            classification_error.append(sum(rfc_prediction != test_data_labels))  # Error Calculation

    # Find the minimum classification error
    rfc_classification_error.append(min(classification_error))
    print(f"{attributes - iteration} retained dimensions: {min(classification_error)}")

    # Find the index of the dimension of the least classification error
    dimension_to_remove = classification_error.index(min(classification_error))
    print(f"Index of removed dimension from the retained dimensions: {dimension_to_remove}")

    # Remove the dimension from the data
    train_data_reduced = np.delete(train_data_reduced, obj=dimension_to_remove, axis=1)
    test_data_reduced = np.delete(test_data_reduced, obj=dimension_to_remove, axis=1)


# Plot the errors
x_axis = [i for i in range(6)]

backward_search_error_plot = plt.figure(1)
plt.plot(x_axis, dtc_classification_error)
plt.plot(x_axis, rfc_classification_error)
plt.xlabel("Retained dimensions out of 6 total dimensions in the dataset")
plt.ylabel("Number of errors")
plt.legend(["Decision Tree Classification Errors", "Random Forest Classification Errors"])
plt.xticks(ticks=[i for i in range(6)], labels=[i for i in range(6, 0, -1)])
plt.grid()
plt.title("Plot of Classification errors after dimensionality reduction using Backward Search")
backward_search_error_plot.show()

# Record end time and show total run time
print(f"\nProgram completed in {(datetime.now() - start_time).seconds}."
      f"{(datetime.now() - start_time).microseconds} seconds.")

plt.show()

# --- Decision Tree Classification and Random Forest Classification using backward search ends ---
