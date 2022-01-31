import csv
import matplotlib.pyplot as plt
import MovingWindow as mw
import Seasonal as seasonal
import KNN as knn


class AnalyzeData:
    def __init__(self):
        self.rows = []
        self.stationList = {}
        self.months = {}

    def main(self):
        self.openFile()
        self.cleanCVS()
        #self.movingWindow()
        self.seasonal()
        #self.KNN()

    def openFile(self):
        file = open('F33GD2.csv', 'r')
        type(file)

        csvreader = csv.reader(file)
        header = next(csvreader)

        for row in csvreader:
            self.rows.append(row)

        file.close()

    def cleanCVS(self):
        for row in list(self.rows):
            if row[5] == '0':
                self.rows.remove(row)
                continue

            stationKeys = self.stationList.keys()

            if stationKeys.__contains__(row[0]):
                self.stationList[row[0]].append([row[2], row[3], row[4]])
            else:
                self.stationList[row[0]] = [[row[2], row[3], row[4]]]

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

    def yearlyDegree(self):
        keys = self.stationList.keys()
        degrees = {}
        months = [['1', 1], ['12', 31]]

        for key in keys:
            for j in range(2007, 2021):
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

    def movingWindow(self):
        mw.MovingWindow(self.rows).flow()

    def seasonal(self):
        seasonal.SeasonalAdjustment(self.stationList).flow()

    def KNN(self):
        knn.KNNModel(self.stationList).flow()


if __name__ == '__main__':
    analyze = AnalyzeData()
    analyze.main()
