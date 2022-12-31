import numpy as np  #create numpy array(python list)
import pandas as pd  # create data frame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading csv data to pandas data frame
heart_data = pd.read_csv("heart_data.csv")
print(heart_data.head(10))  #printing first 10 rows of heart dataset
print(heart_data.tail(10))  #printing last 10 rows of heart dataset

print(heart_data.shape)  #number of rows and coloumn
print(heart_data.info())  #gives info about data

print(heart_data.isnull().sum(
))  #isnull() finds number of missing values and sum() to find for all coloumn

print(heart_data.describe())  #stastical measure about the data

print(
    heart_data['target'].value_counts()
)  #checking the distribution of num variable i.e. counting the number of 0,1 in variable target
"""
0 ---> no heart disease
1 ---> suffering from heart disease 
"""

# now splitting the features and target
x = heart_data.drop(
    columns='target',
    axis=1)  #while dropping coloumn, axis = 1 and while dropping row, axis = 0
y = heart_data['target']
print(x)
print(y)

#splitting data into training and test data
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=2)

print(x.shape, x_train.shape, x_test.shape)

# model training
#LogisticRegression
model = LogisticRegression()

# training the logistic regression model with training data
model.fit(x_train, y_train)

# model evaluation
# accuaracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("Accuracy on training data:", training_data_accuracy)

# accuaracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Accuracy on test data:", test_data_accuracy)

#building the predictive System
input_data = (63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1)

#changing input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshaping the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)

# result(prediction)
# result(prediction)
if (prediction[0] == 0):
    print("The person does not have a heart disease")
else:
    print("The person is suffering from heart disease.")

  # model_ecport
import pickle
model1_pkl = "model.pkl"
with open('model1_pkl', 'wb') as files:
    pickle.dump(model, files)
