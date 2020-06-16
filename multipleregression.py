from sklearn import linear_model
import pandas as pd

def get_rsquared_coefficient(dependentVariable, *datapoints):
    columnHeaders = []
    for i in range(0, len(datapoints) + 1):
        columnHeaders.append(str(i))

    allData = []
    for i in range(0, len(dependentVariable)):
        listToAdd = []
        listToAdd.append(dependentVariable[i])

        for independantVariable in datapoints:
            listToAdd.append(independantVariable[i])

        allData.append(listToAdd)

    df = pd.DataFrame(data=allData, columns=columnHeaders)
    X = df[columnHeaders[1:]]
    y = df["0"]

    model = linear_model.LinearRegression()
    model.fit(X, y)
 
    return [model.score(X, y), 1 - (1-model.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1)]
