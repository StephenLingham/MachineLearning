from sklearn.ensemble import RandomForestClassifier
import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import numpy as np
import types
from sklearn import metrics

from sklearn.model_selection import train_test_split

#import csv
df = pandas.read_csv("./train.csv")

# drop non useful columns


df = df.drop(columns=["PassengerId", "Name", "Ticket",
                      "Fare", "Cabin"])

# drop rows with any empty cells
# https://hackersandslackers.com/pandas-dataframe-drop
df.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)
# One-hot encode the data using pandas get_dummies
# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
df = pandas.get_dummies(df)

print(df.head())
print(df.dtypes)


# split data into "features" and "targets" aka "features" and "label"
# Labels are the values we want to predict, features the variables we use to predict them


# Use numpy to convert column to array of values
labels = np.array(df['Survived'])

# Remove the label column from the df
df = df.drop('Survived', axis=1)

# Saving list of feature names as numpy array
features = np.array(df)

# use train_test_split to divide the data into train and test data
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42)

# check eveything looks right
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators=1000, random_state=42)
# Train the model on training data
rf.fit(train_features, train_labels)

# generate predictions for the test data based on test features
predictions = rf.predict(test_features)

# just compare length of the test_labels with the length of the predictions to make sure they are they same
print(len(test_labels))
print(len(predictions))

# compare each actual result on the test_label list and the predicted list and populate true or false depending on if the prediction was right
results = []
for i in test_labels:
    if test_labels[i] == predictions[i]:
        results.append("TRUE")
    else:
        results.append("FALSE")

print(results)


# create dataframe of our results
dictionaryForDataFrame = {"Predicted Outcome": predictions,
                          "Actual Outcome": test_labels, "Prediction Successful": results}

resultsDataFrame = pandas.DataFrame(dictionaryForDataFrame)

print(resultsDataFrame)
# looks like it is pretty accurate but is there any wrong results? check if any 'falses'
print(results.count("FALSE"))
