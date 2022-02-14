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
        self.x2021 = list()
        self.y2021 = list()

    def flow(self):
        self.KNN()

    def setupDataFrame(self, station):
        degreeDay = []
        date = []
        for item in self.stationsList[station]:
            degreeDay.append(float(item[1]))
            splitDate = str(item[0]).split('-')
            dayOfYear = datetime.date(int(splitDate[0]), int(splitDate[1]), int(splitDate[2])).timetuple().tm_yday
            date.append(dayOfYear)
            # date.append(datetime.datetime.fromisoformat(item[0]))

        data = {'degree_days': degreeDay,
                'dates': date}

        self.data[station] = pd.DataFrame(data)

    # Init of training and testing sets
    def KNN(self):
        for station in self.stationsList.keys():
            if station != '102008':
                continue

            self.setupDataFrame(station)
            self.x = self.data[station].drop("degree_days", axis=1)
            self.x = np.array(self.x['dates']).reshape(-1, 1)
            self.y = self.data[station].drop("dates", axis=1)
            self.y = self.y.values

            length = len(self.y)
            lengthOf2021 = 0
            for i in range(length - 1, 0, -1):
                if self.x[i] == 1:
                    lengthOf2021 += 1
                    break
                else:
                    lengthOf2021 += 1

            self.x2021 = list()
            self.y2021 = list()

            for i in range(length - lengthOf2021, length):
                self.x2021.append(self.x[i])
                self.y2021.append(self.y[i])

            self.x2021 = np.array(self.x2021).reshape(-1, 1)
            # Split into training and test set
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.x, self.y, test_size=0.35, random_state=123456)

            self.testKValues()
            self.findBestKValue()
            self.calculateKWithWeights()

    # Tests the values between 1 and 15,
    # to check which one has the best RMSE and R²
    def testKValues(self):

        scores = list()
        score = list()
        x = list()
        for i in range(1, 50):
            knn = KNeighborsRegressor(n_neighbors=i)
            knn.fit(self.X_train, self.y_train)
            score.append(knn.score(self.X_test, self.y_test))
            # print('Score %f' % score[i - 1])

            x.append(i)
            scores.append(self.calculateRMSE(knn, 'K with random K-value (%d), station: %s' % (i, x), False))

        RMSE = list()
        MAE = list()
        for item in scores:
            RMSE.append(item[0])
            MAE.append(item[1])

        plt.subplot(2, 1, 1)
        plt.plot(x, RMSE, label='RMSE')
        plt.plot(x, RMSE, 'o')
        plt.xlabel('K-value')
        plt.ylabel('RMSE')
        plt.title('RMSE for the different K-values')
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.subplots_adjust()
        plt.plot(x, MAE, label='MAE')
        plt.plot(x, MAE, 'o')
        plt.xlabel('K-value')
        plt.title('MAE for the difference K-values')
        plt.ylabel('MAE')
        plt.grid()
        plt.tight_layout()
        plt.show()

    # Finding the best K value by using predictions
    def findBestKValue(self):
        # Parameters for gridSearch
        parameters = {
            "n_neighbors": range(1, 50)}
        gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
        gridsearch.fit(self.X_train, self.y_train)

        print(gridsearch.best_params_)  # Best k-value

        neighbors = str(gridsearch.best_params_)[16:18]

        self.calculateRMSE(gridsearch, 'Finding the best K-value, neighbors: %s' % neighbors)

    # Test if it makes sense to use weighted averages
    def calculateKWithWeights(self):
        # Parameters for gridSearch
        parameters = {
            "n_neighbors": range(1, 50),
            "weights": ["distance"]}
        gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
        gridsearch.fit(self.X_train, self.y_train)

        print(gridsearch.best_params_)  # best K-value

        neighbors = str(gridsearch.best_params_)[16:18]

        self.calculateRMSE(gridsearch, 'Finding best K-value and weights, neighbors: %s' % neighbors)

        print('-------------\n')

    # Calculates the RMSE value and returns it
    def calculateRMSE(self, model, title, shouldPrint=True):

        test_pred = model.predict(self.X_test)
        test_pred = model.predict(self.x2021)
        mse = mean_squared_error(self.y2021, test_pred)

        RMSE = np.sqrt(mse)

        sum2021 = 0
        for i in self.y2021:
            sum2021 += i[0]

        predSum = 0
        for degree in test_pred:
            predSum += degree

        MAE = (sum2021 - predSum) / len(self.x2021)

        x_predicted, y_predicted = self.sort(self.x2021, test_pred)
        x_real, y_real = self.sort(self.x, self.y)

        if shouldPrint:
            print(title)
            print('Test RMSE: %f' % RMSE)
            print('MAE: %F' % MAE)

            plt.plot(x_real, y_real, 'o', )
            plt.plot(self.x2021, self.y2021, 'y')
            plt.plot(x_predicted, y_predicted, 'r')
            plt.xlabel('Years')
            plt.ylabel('Degree days')
            plt.title(title)
            plt.show()

        return [RMSE, MAE]

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
