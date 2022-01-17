import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import statistics as st
from operator import itemgetter
from sklearn.metrics import mean_squared_error, r2_score


# TODO prøv selv beregnet akkumuleret
# TODO undersøg kalman filteret
class AnalyzeData:
    def __init__(self):
        if os.path.exists('cleanedCVS.csv'):
            self.fileName = "cleanedCVS.csv"
        else:
            self.fileName = "F33GD.csv"

        self.file = None
        self.header = []
        self.rows = []
        self.stationList = {}
        self.months = {}

    def main(self):
        self.openFile()
        self.cleanCVS()
        self.createNewCVS()
        # self.yearlyDegree()
        # self.linearRegression()
        # self.movingWindowTail()
        self.movingWindowCenter()

    def openFile(self):
        self.file = open(self.fileName, 'r')
        type(self.file)

        csvreader = csv.reader(self.file)
        self.header = next(csvreader)

        for row in csvreader:
            self.rows.append(row)

        self.file.close()

    def cleanCVS(self):
        for row in list(self.rows):
            if float(row[3]) < 1 or \
                    (float(row[3]) > 45 and row[0] != '102008') or \
                    (float(row[3]) > 50 and row[0] == '102008'):
                self.rows.remove(row)

            stationKeys = self.stationList.keys()

            if stationKeys.__contains__(row[0]):
                self.stationList[row[0]].append([row[2], row[3], row[4]])
            else:
                self.stationList[row[0]] = [[row[2], row[3], row[4]]]

            if self.months.__contains__(row[6]):
                for station in self.months[row[6]]:
                    if station.__contains__(row[0]):
                        station[row[0]].append([float(row[3]), float(row[4]), int(row[5]), int(row[7])])
                    else:
                        station[row[0]] = [[float(row[3]), float(row[4]), int(row[5]), int(row[7])]]
            else:
                self.months[row[6]] = [{row[0]: [[float(row[3]), float(row[4]), int(row[5]), int(row[7])]]}]

    def createNewCVS(self):
        with open('cleanedCVS.csv', 'w', encoding='UTF8') as newCVS:
            writer = csv.writer(newCVS)

            writer.writerow(self.header)
            writer.writerows(self.rows)

            print('file created')
            newCVS.close()

    def linearRegression(self):
        months = self.months

        for month in months:
            dates = {}
            station = self.months[month][0]

            for stationId in station:
                x = []
                y = []
                for item in station[stationId]:
                    if dates.__contains__(item[3]):
                        dates[item[3]].append(item[0])
                    else:
                        dates[item[3]] = [item[0]]

                for date in dates:
                    x.append(date)
                    y.append(st.mean(dates[date]))

                self.polyRegression(x, y, stationId, month)

                regression = linear_model.LinearRegression()

                x1 = np.array(x).reshape((-1, 1))
                y1 = np.array(y)
                regression.fit(x1, y1, 3)

                predict = regression.predict(x1)

                print('Linear r² score: ', r2_score(y1, predict))

                # Plot outputs
                plt.scatter(x1, y1, color="black")
                plt.plot(x1, predict, color="blue", linewidth=3)
                _tempStr = month + ' ' + stationId
                plt.xlabel(_tempStr)
                plt.grid(True)

                plt.show()

    @staticmethod
    def polyRegression(x, y, key, month):
        myModel = np.poly1d(np.polyfit(x, y, 3))
        myLine = np.linspace(1, 31)
        print('-------------------')
        print('StationId', key, 'month', month)
        print('poly r² score', r2_score(y, myModel(x)))

        plt.scatter(x, y)
        plt.plot(myLine, myModel(myLine))

    def yearlyDegree(self):
        keys = self.stationList.keys()
        degrees = {}
        months = [['1', 1], ['12', 31]]

        for key in keys:
            i = 2007
            for j in range(i, 2021):
                for item in months:
                    for date in self.months[item[0]][0][key]:
                        if date[2] == j and date[3] == item[1]:
                            if degrees.__contains__(key):
                                if degrees[key].__contains__(j):
                                    degrees[key][j].append(date[1])
                                    break
                                else:
                                    degrees[key][j] = [date[1]]
                            else:
                                degrees[key] = {j: [date[1]]}

        for key in degrees.keys():
            x = []
            y = []
            for year in degrees[key]:
                _temp = degrees[key][year]
                x.append(year)
                y.append(_temp[1] - _temp[0])

            plt.barh(x, y)
            plt.xlabel(key)
            plt.show()

    def movingWindowTail(self):
        months = self.months.keys()

        for month in months:
            for station in self.months[month]:
                for stationId in station:
                    dates = {}
                    years = {}
                    for item in self.months[month][0][stationId]:
                        if dates.__contains__(item[3]):
                            dates[item[3]].append(item[0])
                        else:
                            dates[item[3]] = [item[0]]

                        if years.__contains__(item[2]):
                            years[item[2]].append([item[3], item[0]])
                        else:
                            years[item[2]] = [[item[3], item[0]]]

                    toSortMax = []
                    toSortMin = []

                    average = []

                    for date in dates.keys():
                        toSortMax.append([date, max(dates[date])])
                        toSortMin.append([date, min(dates[date])])
                        average.append([date, st.mean(dates[date])])

                    for year in years:
                        if year % 4 != 0:
                            continue

                        x, y = self.setXY(years[year])
                        plt.plot(x, y, label=str(year))

                    x, yH = self.setXY(toSortMax)
                    x, yL = self.setXY(toSortMin)
                    x, yA = self.setXY(average)

                    averageX, averageY = self.calculateMovingAverageTail(yH, yL)

                    plt.plot(x, yA, label='Average Temp')
                    plt.plot(x, yH, label='Max')
                    plt.plot(x, yL, label='Min')
                    plt.xlabel('Days')
                    plt.ylabel('Degree day')
                    plt.plot(averageX, averageY, label='Moving Average')
                    plt.title(("Station: ", stationId, "Month: ", month))
                    plt.legend()

                    if 1 <= int(month) < 3:
                        plt.show()
                    else:
                        plt.clf()

    def movingWindowCenter(self):
        months = self.months.keys()
        errorT = []
        errorC = []

        for month in months:
            for station in self.months[month]:
                for stationId in station:
                    dates = {}
                    years = {}
                    for item in self.months[month][0][stationId]:
                        if dates.__contains__(item[3]):
                            dates[item[3]].append(item[0])
                        else:
                            dates[item[3]] = [item[0]]

                        if years.__contains__(item[2]):
                            years[item[2]].append([item[3], item[0]])
                        else:
                            years[item[2]] = [[item[3], item[0]]]

                    toSortMax = []
                    toSortMin = []

                    average = []

                    for date in dates.keys():
                        toSortMax.append([date, max(dates[date])])
                        toSortMin.append([date, min(dates[date])])
                        average.append([date, st.mean(dates[date])])

                    '''
                    for year in years:
                        if year % 4 != 0:
                            continue

                        y = self.setXY(years[year])
                        plt.plot(y, label=str(year))
                    '''

                    yH = self.setXY(toSortMax)
                    yL = self.setXY(toSortMin)
                    yA = self.setXY(average)

                    x2, averageY = self.calculateMovingAverageCenter(yH, yL)
                    x1, y1 = self.calculateMovingAverageTail(yH, yL)


                    '''
                    plt.plot(yA, label='Average Temp')
                    plt.xlabel('Days')
                    plt.ylabel('Degree day')
                    plt.plot(x2, averageY, label='Moving AverageCENTER')
                    plt.plot(x1, y1, label='Moving AverageTAIL')
                    plt.title(("Station: ", stationId, "Month: ", month))
                    plt.legend()
                    '''

                    if 1 <= int(month) < 2:
                        #plt.plot(yH, label='Max')
                        #plt.plot(yL, label='Min')
                        plt.title(("Station:%s, Month:%s " % (str(stationId), str(month))))
                        print("Station: ", stationId, "Month: ", month)
                        tail, center = self.makeMovingWindowPrediction(5, yA, y1, averageY)
                        errorC.append(center)
                        errorT.append(tail)
                        print('--------------------------\n')
                        # plt.show()
                    # else:
                    # plt.clf()
        print('mean of ErrorT= %f' % st.mean(errorT))
        print('mean of ErrorC= %f' % st.mean(errorC))

    @staticmethod
    def setXY(sortList):
        sortList.sort(key=itemgetter(0))
        y = []

        for item in sortList:
            y.append(item[1])

        return y

    @staticmethod
    def calculateMovingAverageTail(high, low):
        x = []
        y = []
        movingWindow = 5
        for i in range(movingWindow, len(high)):
            x.append(i)
            windowSum = 0
            for j in range(i, i - movingWindow, -1):
                windowSum += high[j] + low[j]
            y.append(windowSum / (movingWindow * 2))

        return x, y

    @staticmethod
    def calculateMovingAverageCenter(high, low):
        x = []
        y = []
        movingWindow = 5  # Needs to be an odd number
        for i in range(int((movingWindow - 1) / 2), len(high)):
            windowSum = 0
            if i + (movingWindow - 1) / 2 < len(high):
                k = 0
                for j in range(i, i + int((movingWindow - 1) / 2)):
                    if k == 0:
                        windowSum += high[i] + low[i]
                    else:
                        windowSum += high[i + k] + low[i + k]
                        windowSum += high[i - k] + low[i - k]
                    k += 1
                if windowSum > 0:
                    x.append(i)
                y.append(windowSum / (movingWindow + 1))

        return x, y

    @staticmethod
    def makeMovingWindowPrediction(window, average, movingTail,
                                   movingCenter):
        test = [average[i] for i in range(window, len(average))]
        test1 = [average[i] for i in range(int((window - 1) / 2), len(average) - 2)]
        x = [i for i in range(int((window - 1) / 2), len(average) - 2)]

        for i in range(len(average) - window):
            pass
            # print('PredictedT=%f, expected=%f' % (yhatTail[i], average[i]))
            # print('PredictedC=%f, expected=%f' % (yhatCenter[i], average[i]))
            # print('-------------------------------')

        errorTail = mean_squared_error(test, movingTail)
        errorCenter = mean_squared_error(test1, movingCenter)

        print('Test MSE Tail: %.3f' % errorTail)
        print('Test MSE Center: %.3f' % errorCenter)

        # plot
        plt.plot(test, color='b', label='Average')
        plt.plot(movingTail, color='red', label='Tail')
        plt.plot(x, movingCenter, color='y', label='Center')
        plt.show()

        return errorTail, errorCenter


if __name__ == '__main__':
    analyze = AnalyzeData()
    analyze.main()
