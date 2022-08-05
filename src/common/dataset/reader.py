import csv
import json
import jsonlines


class Reader:
    def __init__(self,encoding="utf-8"):
        self.enc = encoding

    def read(self,file):
        with open(file,"r",encoding = self.enc) as f:
            return self.process(f)

    def process(self,f):
        pass


class CSVReader(Reader):
    def process(self,fp):
        r = csv.DictReader(fp)
        return [line for line in r]

class JSONReader(Reader):
    def process(self,fp):
        return json.load(fp)


class JSONLineReader(Reader):
    def process(self,fp):
        data = []
        for line in fp:
            data.append(line)
        print(len(data))
        return data

    def read(self,file):
        with jsonlines.open(file,"r") as f:
            return self.process(f)

