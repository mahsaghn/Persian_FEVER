import argparse
import json
import jsonlines
from tqdm import tqdm

from common.dataset.reader import JSONLineReader
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from retrieval.filter_uninformative import uninformative

parser = argparse.ArgumentParser()
parser.add_argument('db_path', type=str, help='/path/to/fever.db')

args = parser.parse_args()


jlr = JSONLineReader()

docdb = FeverDocDB(args.db_path)

idx = docdb.get_non_empty_doc_ids()
idx = list(filter(lambda item: not uninformative(item),tqdm(idx)))
print(len(idx))
r = SimpleRandom.get_instance()
lines = []
with jsonlines.open("data/fever/test.ns.rand.jsonl", "w") as f:
    with jsonlines.open("data/fever-data/test.jsonl",'r') as f1:
        for line in f1:
            if line["label"] == "NOT ENOUGH INFO":

                for evidence_group in line['evidence']:
                    for evidence in evidence_group:
                        evidence[-2] = idx[r.next_rand(0, len(idx)-1)]
                        evidence[-1] = -1
            lines.append(line)

        f.write_all(lines)

with jsonlines.open("data/fever/dev.ns.rand.jsonl", "w") as f:
    with jsonlines.open("data/fever-data/dev.jsonl",'r') as f1:
        for line in f1:
            if line["label"]=="NOT ENOUGH INFO":
                for evidence_group in line['evidence']:
                    for evidence in evidence_group:
                        evidence[-2] = idx[r.next_rand(0, len(idx)-1)]
                        evidence[-1] = -1
            lines.append(line)

        f.write_all(lines)



with jsonlines.open("data/fever/train.ns.rand.jsonl", "w") as f:
    with jsonlines.open("data/fever-data/train.jsonl",'r') as f1:
        for line in f1:
            if line["label"]=="NOT ENOUGH INFO":
                for evidence_group in line['evidence']:
                    for evidence in evidence_group:
                        evidence[-2] = idx[r.next_rand(0, len(idx)-1)]
                        evidence[-1] = -1
            lines.append(line)

        f.write_all(lines)
