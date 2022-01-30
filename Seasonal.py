import datetime
from math import sqrt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import polyfit
from sklearn.metrics import mean_squared_error


class SeasonalAdjustment:
    def __init__(self, stations):
        self.stationsList = stations
        self.data = {}

    def flow(self):
        self.setupDataFrame()
        # self.plot()
        # self.adjustment()
        self.seasonalAdjustment()
        # self.modeling()

    def setupDataFrame(self):
        for station in self.stationsList.keys():
            degreeDay = []
            date = []
            for item in self.stationsList[station]:
                degreeDay.append(float(item[1]))
                date.append(datetime.datetime.fromisoformat(item[0]))

            data = {'degree_days': degreeDay}

            self.data[station] = pd.DataFrame(data, index=date)

    def plot(self):

        for x in self.data.keys():
            plt.clf()
            self.data[x].plot()
            if x == '102008':
                plt.title('plot, Station: %s' % x)
                plt.xlabel('Years')
                plt.ylabel('Degree day')
                plt.show()

    def adjustment(self):
        for x in self.data.keys():
            if x != '102008':
                continue

            plt.clf()
            X = self.data[x].values
            diff = list()
            days_in_year = 365
            for i in range(days_in_year, len(X)):
                value = X[i] - X[i - days_in_year]
                diff.append(value)
            if x == '102008':
                plt.plot(diff)
                plt.title('adjustment, Station %s' % x)
                plt.xlabel('years')
                plt.ylabel('degree day')
                plt.show()

    def seasonalAdjustment(self):
        plt.clf()
        for x in self.data.keys():
            if x != '102008':
                continue

            y = self.data[x].values

            diff = list()
            days_in_year = 365
            for i in range(days_in_year, len(y)):
                # String to compare date with
                monthDay = str(self.data[x].index[i].month) + '/' + str(self.data[x].index[i].day)
                if monthDay == '2/29':  # Extra day every four years
                    if i - days_in_year * 4 > 0:  # Checks for out of bounds
                        value = y[i] - y[i - days_in_year * 4]
                    else:
                        value = 0  # Set to 0 because of no difference
                    days_in_year += 1  # adds on day for correction
                elif monthDay == '2/28':
                    value = y[i] - y[i - days_in_year]
                    days_in_year = 365
                else:
                    value = y[i] - y[i - days_in_year]

                diff.append(value)

            if x == '102008':
                plt.plot(diff)
                plt.title('Seasonal adjustment noice for station %s' % x)
                plt.xlabel('Days')
                plt.ylabel('Subtracted noice')
                plt.show()

                _temp = list()
                realData = [y[i] for i in range(365, len(y))]

                for i in range(len(y) - 365):
                    value = realData[i] - diff[i]
                    _temp.append(value)

                plt.plot(_temp)
                plt.title('Seasonal adjustment, Station %s' % x)
                plt.xlabel('Days')
                plt.ylabel('Degree days')
                plt.show()

                dayNumber = [i % 365 for i in range(0, len(self.data[x].values) - 365)]

                for degree in [3, 4, 5, 6]:
                    coef = polyfit(dayNumber, _temp, degree)

                    testCurve = list()
                    curve = list()

                    for i in range(len(dayNumber)):
                        value = coef[-1][0]
                        for d in range(degree):
                            value += dayNumber[i] ** (degree - d) * coef[d][0]
                        curve.append(value)

                    # for i in coef:
                    # print(i)
                    '''
                    plt.subplot(2, 2, degree - 2)
                    plt.plot(_temp, label='Adjusted Data')
                    plt.plot(curve, color='r', label='Curve')
                    plt.xlabel('Days')
                    plt.ylabel('Degree days')
                    plt.title('Degree: %d' % degree)
                    plt.tight_layout()
                    '''

                    RMSE = sqrt(mean_squared_error(realData, curve))

                    if len(testCurve) == 0:
                        testCurve = [curve, RMSE, degree]
                    else:
                        if testCurve[1] < RMSE:
                            testCurve = [curve, RMSE, degree]

                    print('RMSE adjustment: %f for degree: %d' % (RMSE, degree))

                plt.show()

                length = len(realData) - 1
                dailyDiffPlot = list()
                numbers = [10, 9, 7, 6, 5, 4, 3, 2, 1, 0]
                for j in numbers:
                    GD = 0
                    predictedGD = 0
                    n = 0
                    for i in range(length - (1 + 365 * j), length - (366 + 365 * j), -1):
                        GD += realData[i][0]
                        predictedGD += curve[i]
                        n += 1

                    dailyDiff = (GD - predictedGD) / n
                    print(
                        '\nGD: %f, predictedGD: %f, diff: %f for degree: %d' % (GD, predictedGD, dailyDiff, degree))
                    print('Daily difference: %f' % dailyDiff)
                    dailyDiffPlot.append(dailyDiff)

                plt.xlabel('Years from 2011')
                plt.ylabel('Daily difference')
                dailyDiffPlot.reverse()

                plt.plot(numbers, dailyDiffPlot)
                plt.show()

                print('\nChosen number of degrees: %d' % testCurve[2])
                _correctForm = []
                for item in _temp:
                    _correctForm.append(item[0])
                ApplyKalmanFilter(curve, realData).foo()

            print('----------- \n')

    def modeling(self):
        for x in self.data.keys():
            if x != '102008':
                continue

            plt.clf()
            dayNumber = [i % 365 for i in range(0, len(self.data[x].values))]
            y = self.data[x].values
            degree = 4
            coef = polyfit(dayNumber, y, degree)

            print('Normal Seasonal')
            for i in coef:
                print(i)

            print('-----------')
            # create curve
            curve = list()

            for i in range(len(dayNumber)):
                value = coef[-1][0]
                for d in range(degree):
                    value += dayNumber[i] ** (degree - d) * coef[d][0]
                curve.append(value)

            # create seasonally adjusted
            diff = list()

            for i in range(len(y)):
                value = y[i] - curve[i]
                diff.append(value)

            _temp = list()

            for i in range(len(dayNumber)):
                value = y[i] - diff[i]
                _temp.append(value)

            if x == '102008':
                plt.plot(diff, color='red', linewidth=1)
                plt.xlabel('days')
                plt.ylabel('Substracted value')
                plt.title('modelling1, Station %s' % x)

                plt.show()

                plt.plot(y)
                plt.plot(_temp, linewidth=3, color='r')
                plt.xlabel('days')
                plt.ylabel('Degree day')
                plt.title('modelling2, Station %s' % x)

                plt.show()

                print('RMSE form: %f' % sqrt(mean_squared_error(y, _temp)))

                _correctForm = []
                for item in _temp:
                    _correctForm.append(item[0])
                ApplyKalmanFilter(_correctForm, y).plot()


class ApplyKalmanFilter:
    def __init__(self, seasonalModel, measurement):
        self.measurement = measurement
        self.model = seasonalModel
        self.x = np.array([seasonalModel[0], 0]).reshape((2, 1))
        self.P = np.eye(2)

    def predict(self, dt: float):
        # x = f x
        # P = F P Ft + G Gt a
        F = np.array([[1, dt], [0, 1]])
        new_x = F.dot(self.x)

        G = np.array([0.5 * dt ** 2, dt]).reshape(2, 1)
        new_p = F.dot(self.P).dot(F.T) + G.dot(G.T) * 1.0

        self.x = new_x
        self.P = new_p

    def plot(self):
        plt.clf()

        covariance = []
        measurements = []

        for i in range(1, len(self.measurement)):
            measurements.append(self.x)
            covariance.append(self.P)

            self.predict(0.3)
            self.update(self.model[i], 5 ** 2)

        plt.title('TEST1')

        plt.plot([self.model[i] for i in range(0, len(self.model))], 'b--')
        plt.plot([a[0] for a in measurements], 'r')

        modelS = 0
        modelR = 0
        _test = []
        for i in range(0, len(self.model) - 1):
            modelR += self.measurement[i]
            modelS += self.model[i]
            _test.append(self.measurement[i][0])

        a = [a[0] for a in measurements]
        modelK = sum(a)
        print('ModelR: %f \nModelK: %f' % (modelR, modelK))

        _test2 = list()
        for item in measurements:
            _test2.append(item[0][0])

        b = mean_squared_error(_test, _test2)
        b = np.sqrt(b)

        print('%f\n' % b)

        plt.show()

    def update(self, measurement, variance):
        # Y = z -H X
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x+ K Y
        # P = (I - K H) * P

        z = np.array([measurement])
        R = np.array([variance])
        H = np.array([1, 0]).reshape((1, 2))

        Y = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self.x + K.dot(Y)
        new_P = (np.eye(2) - K.dot(H)).dot(self.P)

        self.x = new_x
        self.P = new_P

    def foo(self):
        from filterpy.kalman import KalmanFilter
        f = KalmanFilter(dim_x=2, dim_z=1)

        f.x = self.x

        f.F = np.array([[1., 0.5], [0, 1]])

        f.H = np.array([1, 0]).reshape((1, 2))

        f.P = self.P

        f.R = 5

        from filterpy.common import Q_discrete_white_noise
        f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=25)

        covariance = []
        measurements = []
        predicted = list()
        for i in range(1, len(self.measurement)):
            measurements.append(self.x)
            covariance.append(self.P)

            f.predict()
            f.update(self.model[i])

            predicted.append(f.x[0, 0])

        plt.plot(predicted, color='red')
        plt.title('Kalman')
        plt.show()

        modelR = 0
        modelK = 0
        _test = []
        for i in range(0, len(self.model) - 1):
            modelR += self.measurement[i]
            modelK += predicted[i]
            _test.append(self.measurement[i][0])

        print('\n\n\nModelR: %f \nModelK: %f' % (modelR, modelK))
        print('diff: %f' % ((modelR - modelK)/365))

        _test2 = list()
        for item in measurements:
            _test2.append(item[0][0])

        b = mean_squared_error(_test, predicted)
        b = np.sqrt(b)

        print('%f\n' % b)
