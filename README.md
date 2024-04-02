# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics. 10.Find the accuracy of our model and predict the require values. `
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: K SHAKTHI SUNDAR
RegisterNumber:  212222040152
*/

import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Import plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv("/content/Employee.csv")

# Display the first few rows of the data
print(data.head())

# Get information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Check the distribution of the target variable
print(data["left"].value_counts())

# Use LabelEncoder to encode the 'salary' column
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Select features (X) and target variable (y)
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
y = data["left"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Initialize the Decision Tree classifier
dt = DecisionTreeClassifier(criterion="entropy")

# Train the classifier
dt.fit(x_train, y_train)

# Make predictions on the test set
y_pred = dt.predict(x_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict on new data
prediction = dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
print(prediction)

plt.figure(figsize=(18, 6))
plot_tree(dt, feature_names=x.columns, class_names=['LEFT', 'NOT LEFT'], filled=True)  # Fix plot_tree call
plt.show()

```

## Output:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128116143/0690feee-3aad-4654-a479-0fc7ea75004c)

![image](https://github.com/ShakthiSundar-K/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128116143/f81721e4-cd2d-43a3-a584-9fbfe46dc176)

![image](https://github.com/ShakthiSundar-K/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128116143/d0f2d4a0-2269-40bb-a883-78ee08fcd154)

![image](https://github.com/ShakthiSundar-K/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128116143/57c8d94e-ffd4-4588-ae72-70f0ebab6f05)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
