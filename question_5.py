import    os
import    time
import    numpy                      as       np
from      sklearn.model_selection    import   KFold
from      sklearn.svm                import   SVC




# Load dataset
print("Dataset is loading ... \n")
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
    x = np.array(x)  
    return np.where(abs(x) < abs(x - 1), 0, 1)


if __name__ == '__main__':
    start_time = time.time()
    print("SVM ..................\n")
    x = TRAIN_DATA_IMG.reshape(1000,784)
    y = TRAIN_DATA_LBL

    kf = KFold(n_splits=10, shuffle = True)

    accuracies = []
    C_values = np.random.uniform(low=10e-5, high=10e5, size=150)

    print("Finding the best coefficient for SVM ..... \n ")

    for i in C_values:
        acrc = []
        for train_index, val_index in kf.split(x):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model = SVC(C = i, kernel = 'linear')
            model.fit(x_train,y_train)

            predictions = model.predict(x_val)
            acc_rate = accuracy(classificate(predictions),y_val)
            acrc.append(acc_rate)
        
        mean_acrc = np.mean(acrc)
        accuracies.append(mean_acrc)

    # Get the indices of the top 5 accuracies
    best_indices        =    np.argsort(accuracies)[-3:]  
    best_C_values       =    C_values[best_indices]  
    best_accuracies     =    np.array(accuracies)[best_indices]  
    best_index          =    np.argmax(accuracies)

    print("Best Cofficient Values:", best_C_values)
    print("Corresponding Accuracies:", best_accuracies*100, "\n")
    print(f"Best Cofficient Value: {C_values[best_index]:.3f}, Best Accuracy: {accuracies[best_index]*100:.1f}%")

    print(f"Training finished in: {time.time() - start_time:.5f} s \n")

    # Testing with validation dataset B
    print("Start testing with validation dataset B")
    model_2 = SVC(C = best_index, kernel = 'linear')
    model_2.fit(TRAIN_DATA_IMG.reshape(1000,784),TRAIN_DATA_LBL)
    predictions_2 = classificate(model_2.predict(TEST_DATA_IMG.reshape(1000,784)) )


    print(f"Accuracy:  {accuracy(predictions_2,TEST_DATA_LBL)*100:.1f}%\n")

    print(f"Program finished in: {time.time() - start_time:.5f} s, have a good day! ")





    


    



   
    
    