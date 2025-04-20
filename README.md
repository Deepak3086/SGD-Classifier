![image](https://github.com/user-attachments/assets/4de7dd2d-a957-43c1-837d-20d216e02462)# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: DEEPAK JG
RegisterNumber:  212224220019
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()

# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows of the dataset
print(df.head())

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:
![image](https://github.com/user-attachments/assets/e4df8e41-5448-4f41-904d-0230ed160a9e)

![image](https://github.com/user-attachments/assets/d3129087-815a-4430-9e5d-fc61b9d6669f)

![image](https://github.com/user-attachments/assets/a0e2d00a-db65-46ff-bcd7-e7f091b8d6db)

![image](https://github.com/user-attachments/assets/74ec5193-17f4-4cf0-a9c9-c62c0697802c)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
