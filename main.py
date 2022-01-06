import csv
import os


class AnalyzeData:
    def __init__(self):
        if os.path.exists('cleanedCVS.csv'):
            self.fileName = "cleanedCVS.csv"
        else:
            self.fileName = "GRADDAGE_TAL.csv"

        self.file = None
        self.header = []
        self.rows = []

    def main(self):
        self.openFile()

    def openFile(self):
        self.file = open(self.fileName, 'r')
        type(self.file)

        csvreader = csv.reader(self.file)
        self.header = next(csvreader)

        for row in csvreader:
            self.rows.append(row)

        self.file.close()

    def cleanCVS(self):
        print('hello')


if __name__ == '__main__':
    analyze = AnalyzeData()
    analyze.main()
