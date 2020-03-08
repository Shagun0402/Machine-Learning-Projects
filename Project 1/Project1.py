import numpy as np
import math
from sklearn.metrics import accuracy_score
import pandas as pd

# From the given X Data and Y Data, Calculates the value for Beta Vector that corresponds to data in X.
def model_fit(X_value, Y_value):
    inverse = np.linalg.inv(np.dot(X_value.T,X_value))
    product = np.dot(X_value.T, Y_value)
    B_coeff = np.dot(inverse, product)
    return B_coeff

# Approximating Class Value for K-Fold Validation
def approx_class_value(A):
    for i in range(len(A)):
        decimal_value = A[i] - int(A[i])
        if decimal_value < 0.5:
            A[i] = math.floor(A[i])
        elif decimal_value > 0.5:
            A[i] = math.ceil(A[i])
    return A

# From given testX and Beta Value, predicts the Class Value i.e. Y = X . Beta Value
def prediction(X, B_vector):
    return approx_class_value(np.dot(X, B_vector))


"""
From testY and Predicted Y, calculate Residue Error i.e. Residual Error = Test case values - Predicted Values. 
Also calculates root mean square error = Sum of Sqaures of Residual Error / length of values
"""
def residue_error(actual_val, predicted_val):
    residue = []
    total_error = 0.0

    for i in range(len(predicted_val)):
        residue.append(actual_val[i] - predicted_val[i])
        total_error = total_error + pow(residue[i], 2)
    root_mean_square_error = total_error/ float(len(actual_val))
    error_residue = pd.DataFrame(residue, columns= ['Residual Error'])
    return (error_residue, root_mean_square_error)

# Gives percentage accuracy between predicted class label and actual label
def accuracy(actual, predicted):
    return accuracy_score(actual, predicted)*100

#
def Kfold (data,K,classifier):
    accurate_data = []

    if K <= 1:
        print("Choose a K to create folds, preferably more than 1")
        return
    else:
        split = int(len(data)/ K)
        for i in range(1,K+1):
            from_range = split * (i -1)
            to_range = split * i
            train_data = data.iloc[0:from_range, :].append(data.iloc[to_range:, :], ignore_index=True)
            validation_data = data.iloc[from_range : to_range]
            trainX = train_data.iloc[:, :4]
            trainY = train_data.iloc[:, 4:]
            count = model_fit(trainX, trainY)
            testX = validation_data.iloc[:, :4]
            testY = validation_data.iloc[:, 4:]
            testing_prediction = prediction(testX, count)
            accurate_data.append(accuracy(testY, testing_prediction))

    return accurate_data

if __name__ =='__main__':

    Attributes = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']

    # Changing value of attribute 'Class' from String to Numerical
    Class_value = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica':3}

    # Loading Iris Dataset from csv file
    dataset = pd.read_csv ("iris_dataset.csv", names= Attributes, header=None)
    dataset = dataset.replace({'Class': Class_value})
    # Printing Data
    print(dataset)

"""
 Training the model and checking whether it can predict given classes from a random dataset
"""
print("-------------------------------------------------------------------------------")
print("Training the Model")
X_data = dataset.iloc[:, :4]
Y_data = dataset.iloc[:, 4:]
B_value = model_fit(X_data, Y_data)
print("The Beta Coefficient value for the given Iris Dataset is:" ,B_value.T)

print("---------------------------------------------------------------------------------")
"""
Testing prediction function using the above calculated B_value to compare between Predicted and Actual Data
"""
print("----------------------------------------------------------------------------------")
print("Testing the model")
test_data = pd.read_csv("iris_data_test.csv", names=Attributes,header=None)
test_data = test_data.replace({'Class':Class_value})
testX = test_data.iloc[:, :4]
testY = test_data.iloc[:, 4:]

# Predicting Values
predicted_Y = prediction(testX,B_value)
print(" The predicted values of Y are :")
print(predicted_Y.T)

print(" The actual values of Y are :")
print(testY.values.T)


# Calculating Accuracy between Predicted Y and Actual Y
print( " Accuracy from Prediction between predicted and Actual Values are:")
print(' {} %' .format(accuracy(testY, predicted_Y)))

residual_error, root_mean_sq_value = residue_error(testY.values, predicted_Y)
print(" Root Mean Squared Value : ")
print(root_mean_sq_value)
print (" Residual Error Value :")
print(residual_error.T)
print("--------------------------------------------------------------------------------------")


# K- Fold Cross Validation

print("----------------------------------------------------------------------------------------")
print("Performing K- Fold Cross Validation")
print("Using K = 5")
data_shuffle = dataset.sample(frac=1)
array = Kfold(data_shuffle,5, model_fit)
print("The accuracy array after KFold Validation is :")
print(array)
print("Average Accuracy after cross-validation:")
print('{}% ' .format(sum(array)/len(array)))
print("---------------------------------------------------------------------------------------------------")






