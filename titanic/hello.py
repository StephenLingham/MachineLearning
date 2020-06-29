import pandas
from sklearn import metrics
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.model_selection import train_test_split
# df = pandas.read_csv("./comedianData.csv")
# dfDiabetes = pandas.read_csv("./diabetesData.csv")
# zoo = pandas.read_csv(".//practice data/datasets_586_1114_zoo.csv")

# # this is just cleaning the comedian data
# d = {'UK': 0, 'USA': 1, 'N': 1}
# df['Nationality'] = df['Nationality'].map(d)
# d = {'YES': 1, 'NO': 0}
# df['Go'] = df['Go'].map(d)

# # the function won't accept any non-numerical data
# del zoo["animal_name"]

# creates a decision tree and tests it using one data set (i.e. split the data set, trains a decision tree on one part and tests it on the other)


def createDecisionTreeAndTest(targetValue, dataFrame):
    features = list(dataFrame.columns)
    indexOfTarget = features.index(targetValue)
    del features[indexOfTarget]

    X = dataFrame[features]
    y = dataFrame[targetValue]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X_train, y_train)
    data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(data)
    graph.write_png('mydecisiontree.png')
    img = pltimg.imread('./mydecisiontree.png')
    imgplot = plt.imshow(img)
    plt.show()
    y_pred = dtree.predict(X_test)
    return ({"Accuracy": metrics.accuracy_score(y_test, y_pred)})


# createDecisionTreeAndTest("Outcome", dfDiabetes)
# createDecisionTreeAndTest("class_type", zoo)


'''


# simply creates a decision tree from a dataframe given a target value in the form of a string which is the target value column heading

def createDecisionTree(targetValue, dataFrame):
    features = list(dataFrame.columns)
    indexOfTarget = features.index(targetValue)
    del features[indexOfTarget]

    X = dataFrame[features]
    y = dataFrame[targetValue]

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)
    data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(data)
    graph.write_png('mydecisiontree.png')
    img = pltimg.imread('./mydecisiontree.png')
    imgplot = plt.imshow(img)
    plt.show()
    return dtree


savedDTree = createDecisionTree("Outcome", dfDiabetes)


# teakes a trained decision tree, target value and test data, prints accuracy score

def testTrainedDecisionTreeAccuracy(decisionTree, targetValue, testDataFrame):
    features = list(testDataFrame.columns)
    indexOfTarget = features.index(targetValue)
    del features[indexOfTarget]

    X_test = testDataFrame[features]
    y_test = testDataFrame[targetValue]

    y_pred = decisionTree.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


testTrainedDecisionTreeAccuracy(savedDTree, "Outcome", dfDiabetes)


# dfDiabetes = pandas.read_csv("./diabetesData.csv")

# features = ['Pregnancies', 'Glucose', 'BMI', 'BloodPressure']

# X = dfDiabetes[features]
# y = dfDiabetes['Outcome']


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=1)

# dtree = DecisionTreeClassifier()
# dtree = dtree.fit(X_train, y_train)
# data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
# graph = pydotplus.graph_from_dot_data(data)
# graph.write_png('mydecisiontree.png')


# img = pltimg.imread('./mydecisiontree.png')
# imgplot = plt.imshow(img)
# # plt.show()


# y_pred = dtree.predict(X_test)

# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

'''
