import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


class KNNModel:
    def __init__(self, stations):
        self.data = {}
        self.stationsList = stations
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def flow(self):
        self.setupDataFrame()
        self.KNN()

    def setupDataFrame(self):
        for station in self.stationsList.keys():
            degreeDay = []
            date = []
            for item in self.stationsList[station]:
                degreeDay.append(float(item[1]))
                date.append(datetime.datetime.fromisoformat(item[0]))

            data = {'degree_days': degreeDay,
                    'dates': date}

            self.data[station] = pd.DataFrame(data)

    def KNN(self):

        for station in self.data.keys():
            if station == '102008':
                x = self.data[station].drop("degree_days", axis=1)
                x = x.values
                y = self.data[station].drop("dates", axis=1)
                y = y.values

                # Split into training and test set
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    x, y, test_size=0.20, random_state=12345)

                self.randomKValue()
                self.findBestKValue()
                self.calculateKWithWeights()
                self.KNNWIthBagging()

    # Tests a random value in this case 3,
    # what the RSME will be from that value
    def randomKValue(self):
        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(self.X_train, self.y_train)

        self.calculateRMSE(knn, 'K with random K-value (3)')

    # Finding the best K value by using predictions
    def findBestKValue(self):
        parameters = {
            "n_neighbors": range(1, 50)}
        gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
        gridsearch.fit(self.X_train, self.y_train)

        print(gridsearch.best_params_)  # bedst paramter (k størrelse)

        self.calculateRMSE(gridsearch, 'Finding best K-value')

    # Finding the K value while using bagging
    def KNNWIthBagging(self):
        parameters = {
            "n_neighbors": range(1, 50),
            "weights": ["uniform", "distance"]}
        gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
        gridsearch.fit(self.X_train, self.y_train)
        best_k = gridsearch.best_params_["n_neighbors"]
        best_weights = gridsearch.best_params_["weights"]

        bagged_knn = KNeighborsRegressor(
            n_neighbors=best_k, weights=best_weights)

        from sklearn.ensemble import BaggingRegressor
        bagging_model = BaggingRegressor(bagged_knn, n_estimators=100)
        bagging_model.fit(self.X_train, self.y_train)

        self.calculateRMSE(bagging_model, 'Bagging model')

    # Test if it makes sense to use weighted averages
    def calculateKWithWeights(self):
        parameters = {
            "n_neighbors": range(1, 50),
            "weights": ["uniform", "distance"]}
        gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
        gridsearch.fit(self.X_train, self.y_train)

        print(gridsearch.best_params_)  # bedst paramter (k størrelse)

        self.calculateRMSE(gridsearch, 'Finding best K-value and weights')

    def calculateRMSE(self, model, title):
        print(title)
        test_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, test_pred)
        rmse = np.sqrt(mse)
        print('Test rmse: %f' % rmse)
        print('-------------\n')

        a, b = self.sort(self.X_test, test_pred)
        c, d = self.sort(self.X_test, self.y_test)
        plt.plot(c, d, linewidth=3)
        plt.plot(a, b, color='r')
        plt.xlabel('Years')
        plt.ylabel('Degree days')
        plt.title(title)
        plt.show()

    @staticmethod
    def sort(x, y):

        temp = list()

        for i in range(0, len(x)):
            temp.append([x[i][0], y[i]])

        temp.sort()

        _tempX = list()
        _tempY = list()

        for i in range(0, len(temp)):
            _tempX.append(temp[i][0])
            _tempY.append(temp[i][1])

        x = np.array(_tempX).reshape((-1, 1))
        y = np.array(_tempY).reshape((-1, 1))

        return x, y
