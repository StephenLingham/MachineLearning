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

# function for extacting digits from ticket number


def newTicket(string):
    digits = [int(s) for s in string.split() if s.isdigit()]
    return digits

# drop non useful columns


df = df.drop(columns=["PassengerId", "Name", "Ticket",
                      "Fare", "Cabin", "Ticket"])

print(df.dtypes)

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

# Use numpy to convert to arrays
# Labels are the values we want to predict
labels = np.array(df['Survived'])
# Remove the labels from the features
# axis 1 refers to the columns
df = df.drop('Survived', axis=1)
# Saving feature names for later use
feature_list = list(df.columns)
# Convert to numpy array
features = np.array(df)

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

predictions = rf.predict(test_features)

print(len(test_labels))


print(len(predictions))

results = []
for i in test_labels:
    if test_labels[i] == predictions[i]:
        results.append("TRUE")
    else:
        results.append("FALSE")

print(results)
