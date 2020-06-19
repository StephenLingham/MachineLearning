import pandas
from sklearn import metrics
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.model_selection import train_test_split


dfDiabetes = pandas.read_csv("./practice data/diabetesData.csv")

features = ['Pregnancies', 'Glucose', 'BMI', 'BloodPressure']

X = dfDiabetes[features]
y = dfDiabetes['Outcome']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')


img = pltimg.imread('./mydecisiontree.png')
imgplot = plt.imshow(img)
# plt.show()


y_pred = dtree.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
