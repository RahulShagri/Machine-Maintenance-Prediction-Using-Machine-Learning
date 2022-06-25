import numpy as np
from sklearn.model_selection import train_test_split

"""Data is split into training and testing data. 25% of the data is reserved for testing."""

# Load original data
data = np.loadtxt("dataset\\ai4i2020.csv", delimiter=",")

# Split the data into training and testing data
data_train, data_test = train_test_split(data, test_size=0.25, random_state=42)

# Calculate the size of the training and testing data
print(f"Training data length: {np.shape(data_train)[0]}")
print(f"Testing data length: {np.shape(data_test)[0]}")

# Save the data
np.savetxt("dataset\\ai4i2020_train_data.csv", data_train, delimiter=",")
np.savetxt("dataset\\ai4i2020_test_data.csv", data_test, delimiter=",")
