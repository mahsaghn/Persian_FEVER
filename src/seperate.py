import argparse

# import torch
import os
import numpy as np

from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
import jsonlines


if __name__ == "__main__":
    # with jsonlines.open("data/fever-data/all.jsonl","a") as f2:
    #     with jsonlines.open("data/fever-data/train.jsonl","r") as f:
    #         for line in f:
    #             f2.write(line)
    #     print("train done")
    #     with jsonlines.open("data/fever-data/dev.jsonl","r") as f:
    #         for line in f:
    #             f2.write(line)
    #     print("def done")
    #     with jsonlines.open("data/fever-data/test.jsonl","r") as f:
    #         for line in f:
    #             f2.write(line)
    #     print("test done")
    data = []
    with jsonlines.open("data/fever-data/all.jsonl","r") as f:
        for line in f:
            data.append(line)
    data = np.array(data)
    data = np.random.choice(data,500)
    with jsonlines.open("data/fever-data/all500.jsonl","w") as f:
        for line in data:
            f.write(line)

    
    

    