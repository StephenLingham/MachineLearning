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

# change non numerical categorisable data into numbers

d = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(d)
d = {'S': 0, 'C': 1, 'Q': 2}
df['Embarked'] = df['Embarked'].map(d)


# function for extacting digits from ticket number

def newTicket(string):
    digits = [int(s) for s in string.split() if s.isdigit()]
    return digits


ticketDigits = []
for ticket in df['Ticket']:
    digit = newTicket(ticket)

    if len(digit) > 0:
        [newDigit] = digit
        ticketDigits.append(newDigit)
    else:
        ticketDigits.append(None)

# add purely numerical ticket number column

df["TicketsClean"] = ticketDigits

# add cabin letter column

cabinLetter = []

for cabin in df['Cabin']:
    if isinstance(cabin, float):

        cabinLetter.append(None)
    elif isinstance(cabin, str):

        cabinLetter.append(cabin[0])

df["Cabin Letter"] = cabinLetter

# save 'cleaned' df to new csv (for using in R)
# df.to_csv('titanicTrainingClean.csv')

# print(df.head())
# checking data types of all the columns before going on to do any decision tree stuff

print(df.dtypes)
# bringing in Stephens code for dropping unwanted columns - anything not a float or int or anythingand getting rid of any NA values

df = df.drop(columns=["PassengerId", "Name", "Ticket",
                      "Fare", "Cabin", "Cabin Letter"])

# print(df.head())
# print(df.count)  # 891 rows

# drop rows with any empty cells
# https://hackersandslackers.com/pandas-dataframe-drop
df.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)

# print(df.count)  # 708 rows

# features = ['Pclass', 'Sex', 'Embarked']  # 0.76 accuracy
# features = ['Pclass', 'Sex']  # 0.769
# features = ['Pclass', 'Sex', 'SibSp'] # 0.79
# features = ['Pclass', 'Sex', 'SibSp', 'Parch']  # 0.72
# features = ['Embarked']  # 0.65
# features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'] #0.73
features = ['Pclass', 'Sex', 'SibSp', 'Parch',
            'Embarked', 'TicketsClean']  # 0.81

# features = ['SibSp', 'Parch', 'TicketsClean']  # 0.57


X = df[features]
y = df['Survived']

# create test and train subsets of the data
# https://towardsdatascience.com/machine-learning-basics-descision-tree-from-scratch-part-ii-dee664d46831

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


# make a decision tree based on the training subset

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)

# create visual representation of tree as png
# data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
# graph = pydotplus.graph_from_dot_data(data)
# graph.write_png('titanicdecisiontree.png')

# # display png
# img = pltimg.imread('titanicdecisiontree.png')
# imgplot = plt.imshow(img)
# plt.show()

# create a set of predictions for the testing subset
print('predicting...')
y_pred = dtree.predict(X_test)

# print accuracy comparison for prediction verses actual values in test group
print("getting accuracy...")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# could do with a function which runs every permutation through the process and comes out with the most accurate one
