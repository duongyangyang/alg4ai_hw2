import    os
import    time
import    numpy                      as       np
import    tensorflow                 as       tf
from      sklearn.model_selection    import   KFold

# Load dataset
print("Dataset is loading .................. \n")
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

class RidgeRegression:
    def __init__(self,k):
        self.k = k

    def fit(self, X, y):
        # Flatten images
        X_flat = X.reshape(X.shape[0], -1)  
        y_flat = y.reshape(-1, 1) 

        # Convert data to tensors
        X_tensor = tf.convert_to_tensor(X_flat, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y_flat, dtype=tf.float32)

        # Add a column of ones for the bias term
        X_b = tf.concat([tf.ones((X_tensor.shape[0], 1), dtype=tf.float32), X_tensor], axis=1)

        # Calculate weights using the least squares formula
        self.weights = tf.linalg.inv(tf.matmul(tf.transpose(X_b), X_b) + self.k * tf.eye(X_tensor.shape[1]+1)) @ tf.matmul(tf.transpose(X_b), y_tensor)


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
    print("Ridge Regression ....................\n")

    x = TRAIN_DATA_IMG
    y = TRAIN_DATA_LBL

    kf = KFold(n_splits=10, shuffle = True)

    accuracies = []
    lambda_values = np.random.uniform(low=1e-9, high=1e9, size=150)
    print("Finding the best coefficient for Ridge ..... \n ")

    for i in lambda_values:
        acrc_part = []
        for train_index, val_index in kf.split(x):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Create model and fit
            model = RidgeRegression(i)
            model.fit(x_train, y_train)

            predictions = model.predict(x_val) 

            acrc_part.append(accuracy(predictions,y_val))
        
        mean_acrc = np.mean(acrc_part)
        accuracies.append(mean_acrc)
        
    

    # Get the indices of the top 5 accuracies
    best_indices             =     np.argsort(accuracies)[-3:]  # Get indices of the top 5 accuracies
    best_lambda_values       =     lambda_values[best_indices]  # Get the corresponding Ridge values
    best_accuracies          =     np.array(accuracies)[best_indices]  # Get the top accuracies
    best_index               =     np.argmax(accuracies)

    print("Best Cofficient Values:", best_lambda_values)
    print("Corresponding Accuracies:", best_accuracies*100,"\n")
    print(f"Best Cofficient Value: {lambda_values[best_index]:.3f}, Best Accuracy: {accuracies[best_index]*100:.1f}%")

    print(f"Training finished in: {time.time() - start_time:.5f} s \n")

    # Testing with validation dataset B
    print("Testing with validation dataset B")
    model_2 = RidgeRegression(lambda_values[best_index])
    model_2.fit(TRAIN_DATA_IMG,TRAIN_DATA_LBL)
    predictions_2 = model_2.predict(TEST_DATA_IMG) 
    print(f"Accuracy:  {accuracy(predictions_2,TEST_DATA_LBL)*100:.1f}%\n")


    print(f"Program finished in: {time.time() - start_time:.5f} s, have a good day! ")


    



   
    
    