import statistics as st
from math import sqrt
from operator import itemgetter

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class MovingWindow:
    def __init__(self, rows):
        self.rows = rows
        self.months = {}
        self.MSE = list()
        self.MAE = list()
        self.centerMSE = {}
        self.tailMSE = {}

    def flow(self):
        self.cleanFile()
        self.movingWindow()

    def cleanFile(self):
        for row in list(self.rows):
            if row[5] == '0' or \
                    float(row[3]) < 1 or \
                    (float(row[3]) > 50 and row[0] != '102008') or \
                    (float(row[3]) > 55 and row[0] == '102008') or \
                    row[5] == '0':
                self.rows.remove(row)
                continue

            if self.months.__contains__(row[6]):
                if self.months[row[6]].__contains__(row[0]):
                    if self.months[row[6]][row[0]].__contains__(row[5]):
                        self.months[row[6]][row[0]][row[5]].append([float(row[3]), float(row[4]), int(row[7])])
                    else:
                        self.months[row[6]][row[0]][row[5]] = [[float(row[3]), float(row[4]), int(row[7])]]
                else:
                    self.months[row[6]][row[0]] = {row[5]: [[float(row[3]), float(row[4]), int(row[7])]]}
            else:
                self.months[row[6]] = {row[0]: {row[5]: [[float(row[3]), float(row[4]), int(row[7])]]}}

    def movingWindow(self):
        for month in self.months.keys():
            if '5' >= month < '9':
                for stationId in self.months[month]:
                    if stationId == '102008':
                        years = list()
                        compare = list()

                        for year in self.months[month][stationId]:
                            if self.checkMonthsForStations(stationId, int(year)):
                                continue

                            dates = list()

                            for item in self.months[month][stationId][year]:
                                dates.append([item[2], item[0]])

                            y = self.setXY(dates)

                            if year != '2021':
                                years.append(y)
                            else:
                                compare = y

                        length = len(compare)

                        for x in years:
                            if len(x) < length:
                                length = len(x)

                        windowSize = 5
                        xT, yT = self.calculateMovingAverageTail(years[0], years[1], length, windowSize)
                        xC, yC = self.calculateMovingAverageCenter(years[0], years[1], length, windowSize)

                        plt.title('Station %s, month %s' % (stationId, month))
                        self.movingWindowPrediction(windowSize, yT, yC, compare, length)

        tail = list()
        center = list()
        for item in self.MSE:
            tail.append(item[0])
            center.append(item[1])

        print('Tail RMSE: %f ' % st.mean(tail))
        print('Center RMSE: %f' % st.mean(center))
        print('---------------------')


        tail2 = list()
        center2 = list()

        for item in self.MAE:
            tail2.append(item[0])
            center2.append(item[1])

        print('Tail MAE: %f ' % st.mean(tail2))
        print('Center MAE: %f' % st.mean(center2))
        print('---------------------')

    @staticmethod
    def setXY(sortList):
        sortList.sort(key=itemgetter(0))
        y = []

        for item in sortList:
            y.append(item[1])

        return y

    @staticmethod
    def calculateMovingAverageTail(year1, year2, length, window):
        x = []
        y = []

        movingWindow = window
        for i in range(window - 1, length):
            x.append(i + 1)
            windowSum = 0

            for j in range(i, i - movingWindow, -1):
                windowSum += year1[j] + year2[j]

            y.append(windowSum / (movingWindow * 2))

        return x, y

    @staticmethod
    def calculateMovingAverageCenter(year1, year2, length, window):
        x = []
        y = []
        movingWindow = window  # Needs to be an odd number
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

    def movingWindowPrediction(self, window, tail,
                               center, compare, length):
        test = [compare[i] for i in range(window - 1, length)]
        test1 = [compare[i] for i in range(int((window - 1) / 2), length - int((window - 1) / 2))]
        xT = [i + 1 for i in range(window - 1, length)]
        xC = [i + 1 for i in range(int((window - 1) / 2), length - int((window - 1) / 2))]

        errorTail = mean_squared_error(test, tail)
        errorCenter = mean_squared_error(test1, center)

        self.MSE.append([sqrt(errorTail), sqrt(errorCenter)])

        MAET = []
        MAEC = []
        for i in range(0, len(tail)):
            MAET.append(test[i] - tail[i])
            MAEC.append(test1[i] - center[i])

        self.MAE.append([st.mean(MAET), st.mean(MAEC)])

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
            if year == 2021 or year == 2019 \
                    or year == 2018:
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
            if year == 2021 or year == 2019 \
                    or year == 2017:
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
