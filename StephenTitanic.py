import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

# create dataframe from csv file
df = pandas.read_csv("titanic/train.csv")

d = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(d)

df = df.drop(columns=["PassengerId", "Name", "SibSp",
                      "Parch", "Ticket", "Fare", "Cabin", "Embarked"])

# drop rows with any empty cells
# https://hackersandslackers.com/pandas-dataframe-drop
df.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)
print(df)

features = ['Pclass', 'Sex', 'Age']

X = df[features]
y = df['Survived']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

# create visual representation of tree as png
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('titanicdecisiontree.png')

# display png
img = pltimg.imread('titanicdecisiontree.png')
imgplot = plt.imshow(img)
plt.show()

print(dtree.predict([[3, 1, 10]]))
