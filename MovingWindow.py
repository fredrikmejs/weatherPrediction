import statistics as st
from math import sqrt
from operator import itemgetter

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class MovingWindow:
    def __init__(self, months):
        self.months = months
        self.MSE = list()
        self.centerMSE = {}
        self.tailMSE = {}

    def movingWindow(self):
        self.movingWindowCenter()

    def movingWindowCenter(self):
        windows = [5]

        for windowSize in windows:

            for month in self.months.keys():
                if 5 < int(month) < 9:
                    continue

                for stationId in self.months[month]:
                    #if stationId != '102008':
                        #continue

                    years = list()
                    compare = list()

                    for year in self.months[month][stationId]:
                        if self.checkMonthsForStations(stationId, int(year)):
                            continue

                        dates = list()

                        for item in self.months[month][stationId][year]:
                            dates.append([item[2], item[0]])

                        y = self.setXY(dates)

                        # x = [i for i in range(1, len(y) + 1)]

                        # plt.plot(x, y, label=year)
                        if year != '2021':
                            years.append(y)
                        else:
                            compare = y

                    # plt.xlabel('Days of the month')
                    # plt.ylabel('Degree day')
                    # plt.legend()
                    # plt.show()
                    length = len(compare)

                    for x in years:
                        if len(x) < length:
                            length = len(x)

                    windowSize = windowSize
                    x, y = self.calculateMovingAverageTail(years[0], years[1], length, windowSize)
                    x1, y1 = self.calculateMovingAverageCenter(years[0], years[1], length, windowSize)

                    plt.title('Station %s, month %s' % (stationId, month))
                    self.makeMovingWindowPrediction(windowSize, y, y1, compare, length)

            tail = list()
            center = list()
            for item in self.MSE:
                tail.append(item[0])
                center.append(item[1])

            print('Tail mean: %f ' % st.mean(tail))
            print('Center mean: %f' % st.mean(center))
            print('---------------------')


        _tempC = list()
        _tempT = list()
        x = list()
        for item in self.centerMSE.keys():
            x.append(item)
            _tempC.append(st.mean(self.centerMSE[item]))
            _tempT.append(st.mean(self.tailMSE[item]))

        plt.title('RMSE værdier for 1, 3, 5, 7, 9')
        plt.plot(x, _tempC, label='Center')
        plt.xlabel('WindowSizes')
        plt.ylabel('RMSE value')
        plt.plot(x, _tempT, label='Tail')
        plt.legend()

        plt.show()

    @staticmethod
    def setXY(sortList):
        sortList.sort(key=itemgetter(0))
        y = []

        for item in sortList:
            y.append(item[1])

        return y

    @staticmethod
    def calculateMovingAverageTail(year1, year2, length, windowSize):
        x = []
        y = []

        movingWindow = windowSize
        for i in range(windowSize, length):
            x.append(i + 1)
            windowSum = 0

            for j in range(i, i - movingWindow, -1):
                windowSum += year1[j] + year2[j]

            y.append(windowSum / (movingWindow * 2))

        return x, y

    @staticmethod
    def calculateMovingAverageCenter(year1, year2, length, windowSize):
        x = []
        y = []
        movingWindow = windowSize  # Needs to be an odd number
        startingPoint = int((movingWindow - 1) / 2)

        for i in range(startingPoint, length):
            windowSum = 0

            if i + (movingWindow - 1) / 2 < length:
                k = 0
                for j in range(i, i + startingPoint + 1):
                    if k == 0:
                        windowSum += year1[i] + year2[i]
                    else:
                        windowSum += year1[i + k] + year2[i + k]
                        windowSum += year1[i - k] + year2[i - k]
                    k += 1
                if windowSum > 0:
                    x.append(i + 1)
                y.append(windowSum / (movingWindow * 2))

        return x, y

    def makeMovingWindowPrediction(self, window, tail,
                                   center, compare, length):
        test = [compare[i] for i in range(window, length)]
        test1 = [compare[i] for i in range(int((window - 1) / 2), length - int((window - 1) / 2))]
        xT = [i + 1 for i in range(window, length)]
        xC = [i + 1 for i in range(int((window - 1) / 2), length - int((window - 1) / 2))]

        errorTail = mean_squared_error(test, tail)
        errorCenter = mean_squared_error(test1, center)

        self.MSE.append([sqrt(errorTail), sqrt(errorCenter)])

        if self.centerMSE.keys().__contains__(window):
            self.centerMSE[window].append(sqrt(errorCenter))
            self.tailMSE[window].append(sqrt(errorTail))
        else:
            self.centerMSE[window] = [sqrt(errorCenter)]
            self.tailMSE[window] = [sqrt(errorTail)]

        if window != 5:
            return

        # plot
        plt.plot(compare, color='b', label="2021")
        plt.plot(xT, tail, color='red', label="Tail")
        plt.plot(xC, center, color='y', label="Center")
        plt.legend(loc='best')
        plt.xlabel('days')
        plt.ylabel('Degree days')
        plt.figure(figsize=(7.5, 4.8))
        plt.show()

    @staticmethod
    def checkMonthsForStations(station, year):

        if station == '102008':
            if year == 2021 or year == 2019 or year == 2018:
                return False

        elif station == '102117':
            if year == 2021 or year == 2019 \
                    or year == 2020:
                return False

        elif station == '102206':
            if year == 2021 or year == 2019 \
                    or year == 2017:
                return False

        elif station == '102220':
            if year == 2021 or year == 2012 \
                    or year == 2019:
                return False

        elif station == '102311':
            if year == 2021 or year == 2019 \
                    or year == 2015:
                return False

        elif station == '102322':
            if year == 2021 or year == 2019 \
                    or year == 2017:
                return False

        elif station == '102406':
            if year == 2021 or year == 2019 \
                    or year == 2014:
                return False

        elif station == '102504':
            if year == 2021 or year == 2017 \
                    or year == 2012:
                # or year == 2015 or year == 2013:
                return False

        elif station == '102613':
            if year == 2021 or year == 2017 \
                    or year == 2015:
                return False

        elif station == '102617':
            if year == 2021 or year == 2019 \
                    or year == 2015:
                return False

        elif station == '102714':
            if year == 2021 or year == 2019 \
                    or year == 2017:
                return False

        elif station == '102804':
            if year == 2021 or year == 2019 or year == 2017:
                return False

        elif station == '102810':
            if year == 2021 or year == 2019 \
                    or year == 2020:
                return False

        elif station == '102906':
            if year == 2021 or year == 2019 \
                    or year == 2014:
                return False

        return True
