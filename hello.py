import pandas
from sklearn import metrics
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.model_selection import train_test_split
df = pandas.read_csv("./comedianData.csv")

# # print(df)

# # d = {'UK': 0, 'USA': 1, 'N': 2}
# # df['Nationality'] = df['Nationality'].map(d)
# # d = {'YES': 1, 'NO': 0}
# # df['Go'] = df['Go'].map(d)

# # # print(df)

# # features = ['Age', 'Experience', 'Rank', 'Nationality']

# # X = df[features]
# # y = df['Go']

# # print(X)
# # print(y)

# # dtree = DecisionTreeClassifier()
# # dtree = dtree.fit(X, y)
# # data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
# # graph = pydotplus.graph_from_dot_data(data)
# # graph.write_png('mydecisiontree.png')
# # print(dtree.predict([[40, 10, 7, 1]]))

# # img = pltimg.imread('./mydecisiontree.png')
# # imgplot = plt.imshow(img)
# # plt.show()

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

comedyData3 = pandas.read_csv("./comedianData3.csv")

d = {'UK': 0, 'USA': 1}
comedyData3['Nationality'] = comedyData3['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
comedyData3['Go'] = comedyData3['Go'].map(d)

# print(comedyData3)
features = ['Nationality']

X = comedyData3[features]
y = comedyData3['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')
img = pltimg.imread('./mydecisiontree.png')
imgplot = plt.imshow(img)
print(dtree.predict([[0]]))
# plt.show()
