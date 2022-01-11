import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn import datasets, linear_model
import statistics as st
from sklearn.metrics import mean_squared_error, r2_score


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
        #self.yearlyDegree()
        self.linearRegression()

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
                    if item[2] == 2021:
                        continue

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

    def polyRegression(self, x, y, key, month):
        myModel = np.poly1d(np.polyfit(x, y, 4))
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


if __name__ == '__main__':
    analyze = AnalyzeData()
    analyze.main()
