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
        self.x = list()
        self.y = list()

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

    # Init of training and testing sets
    def KNN(self):
        for station in self.data.keys():
            self.x = self.data[station].drop("degree_days", axis=1)
            self.x = np.array([i for i in range(1, (len(self.x.values) + 1))]).reshape((-1, 1))  # self.x.values
            self.y = self.data[station].drop("dates", axis=1)
            self.y = self.y.values

            # Split into training and test set
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.x, self.y, test_size=0.35, random_state=123456)

            self.testKValues()
            self.findBestKValue()
            self.calculateKWithWeights()

    # Tests the values between 1 and 15,
    # to check which one has the best RMSE and R²
    def testKValues(self):

        RMSE = list()
        score = list()
        x = list()
        for i in range(1, 2):
            knn = KNeighborsRegressor(n_neighbors=2)
            knn.fit(self.X_train, self.y_train)
            score.append(knn.score(self.X_test, self.y_test))
            print(score[i - 1])

            x.append(i)
            RMSE.append(self.calculateRMSE(knn, 'K with random K-value (%d)' % i))

        plt.subplot(2, 1, 1)
        plt.plot(x, RMSE, label='RMSE')
        plt.plot(x, RMSE, 'o')
        plt.xlabel('K-value')
        plt.ylabel('RMSE')
        plt.title('RMSE for the different K-values')
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.subplots_adjust()
        plt.plot(x, score, label='Score')
        plt.plot(x, score, 'o')
        plt.xlabel('K-value')
        plt.title('Score for the difference K-values')
        plt.ylabel('Score')
        plt.grid()
        plt.tight_layout()
        plt.show()

    # Finding the best K value by using predictions
    def findBestKValue(self):
        # Parameters for gridSearch
        parameters = {
            "n_neighbors": range(1, 25)}
        gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
        gridsearch.fit(self.X_train, self.y_train)

        print(gridsearch.best_params_)  # Best k-value

        self.calculateRMSE(gridsearch, 'Finding the best K-value')

    # Test if it makes sense to use weighted averages
    def calculateKWithWeights(self):
        # Parameters for gridSearch
        parameters = {
            "n_neighbors": range(1, 25),
            "weights": ["distance"]}
        gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
        gridsearch.fit(self.X_train, self.y_train)

        print(gridsearch.best_params_)  # best K-value

        self.calculateRMSE(gridsearch, 'Finding best K-value and weights')

    # Calculates the RMSE value and returns it
    def calculateRMSE(self, model, title):
        print(title)

        test_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, test_pred)

        RMSE = np.sqrt(mse)
        print('Test RMSE: %f' % RMSE)

        x_predicted, y_predicted = self.sort(self.X_test, test_pred)
        x_real, y_real = self.sort(self.x, self.y)

        plt.plot(x_real, y_real, linewidth=3)
        plt.plot(x_predicted, y_predicted, color='r')
        plt.xlabel('Years')
        plt.ylabel('Degree days')
        plt.title(title)
        plt.show()

        print('-------------\n')

        return RMSE

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
