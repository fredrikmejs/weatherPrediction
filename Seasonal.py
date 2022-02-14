import datetime
from math import sqrt

import numpy as np
import pandas as pd
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from matplotlib import pyplot as plt
from numpy import polyfit
from sklearn.metrics import mean_squared_error


class SeasonalAdjustment:
    def __init__(self, stations):
        self.stationsList = stations
        self.data = {}
        self.MAE = 0

    def flow(self):
        self.setupDataFrame()
        # self.plot()
        # self.adjustment()
        self.seasonalAdjustment()
        self.modeling()

    def setupDataFrame(self):
        for station in self.stationsList.keys():
            degreeDay = []
            date = []
            for item in self.stationsList[station]:
                degreeDay.append(float(item[1]))
                date.append(datetime.datetime.fromisoformat(item[0]))

            data = {'degree_days': degreeDay}

            self.data[station] = pd.DataFrame(data, index=date)

    # Plots all the data
    def plot(self):
        for x in self.data.keys():
            plt.clf()
            self.data[x].plot()
            plt.title('plot, Station: %s' % x)
            plt.xlabel('Years')
            plt.ylabel('Degree day')
            plt.show()

    # Creates an adjustment
    def adjustment(self):
        for x in self.data.keys():
            plt.clf()
            X = self.data[x].values
            diff = list()
            days_in_year = 365
            for i in range(days_in_year, len(X)):
                value = X[i] - X[i - days_in_year]
                diff.append(value)

    # Creates the model seasonal Adjustment
    def seasonalAdjustment(self):
        plt.clf()
        for x in self.data.keys():
            print('StationID: %s' % x)

            y = self.data[x].values

            diff = self.checkForFebTwentyNine(x, y)

            plt.plot(diff)
            plt.title('Seasonal adjustment noice for station %s' % x)
            plt.xlabel('Days')
            plt.ylabel('Subtracted noice')
            plt.show()

            self.plotSeasonalAdjustment(diff, x, y)

    def checkForFebTwentyNine(self, x, y):
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

        return diff

    def plotSeasonalAdjustment(self, diff, x, y):
        adjustedData = list()
        # Because of the removal of the first 365 days
        realData = [y[i] for i in range(365, len(y))]

        for i in range(len(y) - 365):
            value = realData[i] - diff[i]
            adjustedData.append(value)

        plt.plot(adjustedData)
        plt.title('Seasonal adjustment, Station %s' % x)
        plt.xlabel('Days')
        plt.ylabel('Degree days')
        plt.show()

        curve = self.createCurveSeasonal(adjustedData, realData)

        length = len(realData) - 1
        dailyDiffPlot = list()
        numbers = [10, 9, 7, 6, 5, 4, 3, 2, 1, 0]
        for j in numbers:
            GD = 0
            predictedGD = 0
            n = 0
            for i in range(length - (365 * j), length - (366 + 365 * j), -1):
                GD += realData[i][0]
                predictedGD += curve[0][i]
                n += 1

            dailyDiff = (GD - predictedGD) / n
            dailyDiffPlot.append(dailyDiff)

        plt.xlabel('Years from 2011')
        plt.ylabel('Daily difference')
        plt.title('Seasonal adjustment daily difference over a year, station: %s' % x)
        plt.grid()
        dailyDiffPlot.reverse()
        plt.plot(numbers, dailyDiffPlot)
        plt.plot(numbers, dailyDiffPlot, 'o')
        plt.show()

        diffence = dailyDiffPlot[0]
        print('Daily diff for 2021: %f' % diffence)

        print('\nChosen number of degrees: %d' % curve[2])
        _correctForm = []
        for item in adjustedData:
            _correctForm.append(item[0])
        ApplyKalmanFilter(_correctForm, realData, curve[2]).plot()

    # Creates model from polyfit
    def createCurveSeasonal(self, adjusted, realData):

        curve = self.createSubPlots(adjusted, False, realData)

        return curve

    def createCurveModel(self, data):

        curve = self.createSubPlots(data, True)

        return curve

    def modeling(self):
        for x in self.data.keys():

            plt.clf()
            y = self.data[x].values
            curve = self.createCurveModel(y)
            diff = list()

            for i in range(len(y)):
                value = y[i] - curve[0][i]
                diff.append(value)

            GD = 0
            GD_predicted = 0
            length = len(y) - 1
            for i in range(length - 365, length):
                GD += y[i]
                GD_predicted += curve[0][i]

            difference = GD - GD_predicted
            print('GD: %f, GD_p: %f, different: %f' % (GD, GD_predicted, difference))
            print('Daily difference: %f' % (difference / 365))
            plt.plot(diff, linewidth=1)
            plt.xlabel('days')
            plt.ylabel('Substracted value')
            plt.title('Noice, Station %s' % x)
            plt.show()

            plt.plot(curve[0], linewidth=3)
            plt.xlabel('days')
            plt.ylabel('Degree day')
            plt.title('Model after subtraktion, Station %s' % x)
            plt.show()
            print('RMSE form: %f' % sqrt(mean_squared_error(y, curve[0])))

            _correctForm = []
            for item in curve[0]:
                _correctForm.append(item)
            ApplyKalmanFilter(_correctForm, y, curve[2]).plot()

    def createSubPlots(self, data, isModel, realData=None):
        dayNumber = [i % 365 for i in range(0, len(data))]
        plotNum = 1
        RMSEList = list()
        tempCurve = list()
        for degree in range(3, 15):
            coef = polyfit(dayNumber, data, degree)
            curve = list()

            for i in range(len(dayNumber)):
                value = coef[-1][0]

                for d in range(degree):
                    value += dayNumber[i] ** (degree - d) * coef[d][0]
                curve.append(value)

            plt.subplot(2, 2, plotNum)
            plt.plot(data, label='Adjusted Data')
            plt.plot(curve, color='r', label='Curve')
            plt.xlabel('Days')
            plt.ylabel('Degree days')
            plt.title('Degree: %d' % degree)
            plt.tight_layout()

            if plotNum == 4:
                plotNum = 1
                plt.show()
            else:
                plotNum += 1

            if isModel:
                RMSE = sqrt(mean_squared_error(data, curve))
            else:
                RMSE = sqrt(mean_squared_error(realData, curve))

            RMSEList.append([RMSE, degree])

            if len(tempCurve) == 0:
                tempCurve = [curve, RMSE, degree]
            else:
                if RMSE < tempCurve[1]:
                    tempCurve = [curve, RMSE, degree]

            # print('RMSE adjustment: %f for degree: %d\n' % (RMSE, degree))

        print('The choosen degree is %d' % tempCurve[2])
        plt.show()

        x1 = list()
        y = list()
        for item in RMSEList:
            x1.append(item[1])
            y.append(item[0])

        plt.plot(x1, y)
        plt.plot(x1, y, 'o')
        plt.xlabel('Degrees')
        plt.ylabel('RMSE')
        plt.grid()
        plt.title("Models RMSE for different degrees, station: 102008")
        plt.show()

        return tempCurve


class ApplyKalmanFilter:

    def __init__(self, seasonalModel, measurement, degree):
        self.measurement = measurement
        self.model = seasonalModel
        self.x = np.array([seasonalModel[0], 0]).reshape((2, 1))
        self.P = np.eye(2)
        self.degree = degree

    def plot(self):
        plt.clf()

        measurements = []
        predicted = list()

        f = self.initKalmanFilter()

        for i in range(1, len(self.measurement)):
            measurements.append(self.x)

            f.predict()
            f.update(self.model[i] + np.random.rand() * np.sqrt(0.2 ** 2))

            predicted.append(f.x[0, 0])

        plt.title('Kalman filter')

        plt.plot(self.measurement, label='Measurements')
        plt.plot(predicted, 'r', label='Model from Kalman filter')
        plt.xlabel('Days')
        plt.ylabel('Degree day')
        plt.legend()
        plt.show()

        self.calculateDifferences(predicted)

    def calculateDifferences(self, predicted):

        trueValue = []
        length = len(predicted) - 1
        dailyDiffPlot = list()
        numbers = [10, 9, 7, 6, 5, 4, 3, 2, 1, 0]

        for i in range(0, len(self.measurement) - 1):
            trueValue.append(self.measurement[i][0])

        for j in numbers:
            modelK = 0
            modelR = 0
            n = 0
            for i in range(length - (365 * j), length - (365 + 365 * j), -1):
                modelR += self.measurement[i][0]
                modelK += predicted[i]
                n += 1

            dailyDiff = (modelR - modelK) / n
            dailyDiffPlot.append(dailyDiff)

        print('Average daily difference: %f' % np.mean(dailyDiffPlot))
        MAE2021 = dailyDiffPlot[len(dailyDiffPlot) - 1]
        print('Daily difference for 2021: %f' % MAE2021)

        plt.xlabel('Years from 2011')
        plt.title('Daily difference with Kalman Filter')
        plt.ylabel('Daily difference')
        dailyDiffPlot.reverse()
        plt.plot(numbers, dailyDiffPlot)
        plt.plot(numbers, dailyDiffPlot, 'o')
        plt.grid()
        plt.show()

        RMSE = np.sqrt(mean_squared_error(trueValue, predicted))

        print('RMSE: %f\n' % RMSE)

        self.polyfit(predicted)

        return MAE2021

    def polyfit(self, predicted):
        dayNumber = [i % 365 for i in range(0, len(self.model) - 1)]
        fit = np.polyfit(dayNumber, predicted, self.degree)

        curve = list()

        for i in range(len(dayNumber)):
            value = fit[-1]
            for d in range(self.degree):
                value += dayNumber[i] ** (self.degree - d) * fit[d]
            curve.append(value)

        plt.plot(curve, label='Curve')
        plt.xlabel('Days')
        plt.ylabel('Degree days')
        plt.title('Curve from Kalman filter')
        plt.show()

        # for i in fit:
        # print(i)

    def initKalmanFilter(self):
        kalmanFilter = KalmanFilter(dim_x=2, dim_z=1)

        # First point in the model
        kalmanFilter.x = self.x

        # State transistion matrix
        kalmanFilter.F = np.array([[1., 0.5], [0, 1]])

        # Measurement funktion
        kalmanFilter.H = np.array([1., 0]).reshape((1, 2))

        # Covariance matrix
        kalmanFilter.P = self.P

        # Noice in the measurement, using the RMSE from before
        kalmanFilter.R = 5.5

        kalmanFilter.Q = Q_discrete_white_noise(dim=2, dt=1, var=25)

        return kalmanFilter
