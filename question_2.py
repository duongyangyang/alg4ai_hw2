import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
print("Dataset is loading ...................\n")
CURRENT_DIRECTORY      =    os.getcwd()

TRAIN_DATA_IMG_PATH    =    os.path.join(CURRENT_DIRECTORY, 'Homework2/morpho_mnist/train/train_mnist.npy')
TRAIN_DATA_LBL_PATH    =    os.path.join(CURRENT_DIRECTORY, 'Homework2/morpho_mnist/train/train_label.npy')
TEST_DATA_IMG_PATH     =    os.path.join(CURRENT_DIRECTORY, 'Homework2/morpho_mnist/test/test_mnist.npy')
TEST_DATA_LBL_PATH     =    os.path.join(CURRENT_DIRECTORY, 'Homework2/morpho_mnist/test/test_label.npy')

TRAIN_DATA_IMG         =    np.load(TRAIN_DATA_IMG_PATH)
TRAIN_DATA_LBL         =    np.load(TRAIN_DATA_LBL_PATH)
TEST_DATA_IMG          =    np.load(TEST_DATA_IMG_PATH)
TEST_DATA_LBL          =    np.load(TEST_DATA_LBL_PATH)


def accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels)

def classificate(x):
    x = np.array(x)  # Chuyển đổi đầu vào thành mảng NumPy nếu chưa phải
    return np.where(abs(x) < abs(x - 1), 0, 1)

class LinearRegression:

    def fit(self, X, y):
        # Flatten images
        X_flat = X.reshape(X.shape[0], -1)  
        y_flat = y.reshape(-1, 1) 

        # Convert data to tensors
        X_tensor = tf.convert_to_tensor(X_flat, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y_flat, dtype=tf.float32)

        # Add a column of ones for the bias term
        X_b = tf.concat([tf.ones((X_tensor.shape[0], 1), dtype=tf.float32), X_tensor], axis=1)

        try:
            # Calculate weights using the least squares formula
            self.weights = tf.linalg.inv(tf.matmul(tf.transpose(X_b), X_b)) @ tf.matmul(tf.transpose(X_b), y_tensor)
            print('X is invertible')

        except tf.errors.InvalidArgumentError:
            # If X_b is not invertible, use pseudo-inverse
            X_b_pseudo = np.linalg.pinv(X_b.numpy())  # Use NumPy to compute pseudo-inverse
            self.weights = tf.convert_to_tensor(X_b_pseudo @ y_tensor.numpy(), dtype=tf.float32)  # Convert back to tensor
            print('X\'X is not invertible, caculating by using pseudo inverse of X\n')

    def predict(self, X):
        # Flatten images for prediction
        X_flat = X.reshape(X.shape[0], -1)  # Convert to shape [n_samples, n_features]

        # Convert input to tensor
        X_tensor = tf.convert_to_tensor(X_flat, dtype=tf.float32)

        # Add a column of ones for the bias term
        X_b = tf.concat([tf.ones((X_tensor.shape[0], 1), dtype=tf.float32), X_tensor], axis=1)

        # Make predictions
        predictions = tf.matmul(X_b, self.weights)
        return classificate(predictions.numpy().flatten())  # Convert tensor to numpy array for easier use


if __name__ == '__main__':
    start_time = time.time()

    print("Linear Regression .................... \n")

    # Training data
    x = TRAIN_DATA_IMG
    y = TRAIN_DATA_LBL

    # Create model
    model = LinearRegression()
    model.fit(x, y)

    predictions = model.predict(TEST_DATA_IMG)  

    print(f"Program finished, taking: {time.time() - start_time:.2f} s ")

    print(f"Accuracy:  {accuracy(predictions,TEST_DATA_LBL)*100:.1f}% ")
    


    



   
    
    