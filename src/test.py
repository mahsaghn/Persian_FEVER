import argparse

import torch
import os
import json

import torch.nn.functional as F

from tqdm import tqdm
from common.training.batcher import Batcher, prepare

from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from common.features.feature_function import Features
from common.training.options import gpu
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
from rte.riedel.fever_features import TermFrequencyFeatureFunction
from rte.riedel.model import SimpleMLP
from rte.riedel.sent_features import SentenceLevelTermFrequencyFeatureFunction
from drqa import retriever
from drqa.retriever import DocDB
import jsonlines 
from multiprocessing.pool import ThreadPool
from drqa.retriever import utils
import numpy as np

from drqascripts.retriever.build_tfidf_lines import OnlineTfidfDocRanker
from argparse import Namespace




def tf_idf_sim(claim, lines,freqs=None):
    args = Namespace(tokenizer='simple' , hash_size= int(np.math.pow(2, 24)), ngram=2)
    tfidf = OnlineTfidfDocRanker(args,[line["sentence"] for line in lines],freqs)
    line_ids,scores = tfidf.closest_docs(claim,5)
    ret_lines = []
    for idx,line in enumerate(line_ids):
        ret_lines.append(lines[line])
        ret_lines[-1]["score"] = scores[idx]
    return ret_lines
    # args.tokenizer,
    #         'hash_size': args.hash_size,
    #         'ngram': args.ngram,



def tf_idf_claim(line):
    if 'predicted_pages' in line:
        sorted_p = list(sorted(line['predicted_pages'], reverse=True, key=lambda elem: elem[1]))
        pages = [p[0] for p in sorted_p]
        p_lines = []
        for page in pages:
            page_lines = db.get_doc_lines(page)
            lines= []
            num_line = 0
            for line_p in page_lines.split("\n"):
                try:
                    if len(line_p.split("\t")[1])>1:
                        lines.append(line_p.split("\t"))
                    else:
                        num_line = line_p.split("\t")[0]
                except:
                    if line_p != "" or line_p !=" ":
                        lines.append([num_line,line_p])
            p_lines.extend(zip(lines, [page] * len(lines), range(len(lines))))
        
        lines = []
        for p_line in p_lines:
            lines.append({
                "sentence": p_line[0][1],
                "page": p_line[1],
                "line_on_page": p_line[2]
            })
        scores = tf_idf_sim(line["claim"], lines)
        line["predicted_sentences"] = [(s["page"], s["line_on_page"]) for s in scores]
    return line

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def predict(model, data, batch_size):
    batcher = Batcher(data, batch_size)
    predicted = []
    for batch, size, start, end in batcher:
        d = prepare(batch)
        model.eval()
        logits = model(d).cpu()
        predicted.extend(torch.max(logits, 1)[1])
    return torch.stack(predicted)

def print_evaluation(model,data,ls,log=None):
    features,actual = data
    predicted = predict(model, features, 500).data.numpy().reshape(-1).tolist()[0]
    if predicted == 0:
        return "SUPPORTS"
    elif predicted == 1:
        return "REFUTES"
    return "NOT ENOUGH INFO"


def process(ranker, query, k=1):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    return zip(doc_names, doc_scores)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str, help='db file path')
    parser.add_argument("--sentence", type=str2bool, default=False)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--modeltfidf', type=str, default=None)
    parser.add_argument('--split',type=str)
    parser.add_argument('--count',type=int, default=1)
    args = parser.parse_args()
    db = FeverDocDB(args.db)
    mname = args.model
    k = args.count
    split = args.split
    ranker = retriever.get_class('tfidf')(tfidf_path=args.modeltfidf)

    with jsonlines.open("data/fever-data/{0}.jsonl".format(split),"r") as f:
        with jsonlines.open("{0}.pages.p{1}.jsonl".format(split,k),"w") as f2:
            for line in f:
                pages = process(ranker,line['claim'],k=k)
                line["predicted_pages"] = list(pages)
                line = tf_idf_claim(line)
                f2.write(line)

    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)
    ffns = []
    
    if args.sentence:
        ffns.append(SentenceLevelTermFrequencyFeatureFunction(db, naming=mname))
    else:
        ffns.append(TermFrequencyFeatureFunction(db,naming=mname))


    f = Features(mname,ffns)
    f.load_vocab(mname)

    jlr = JSONLineReader()
    formatter = FEVERGoldFormatter(None, FEVERLabelSchema())

    test_ds = DataSet(file="{0}.pages.p{1}.jsonl".format(split,k), reader=jlr, formatter=formatter)
    test_ds.read()
    feats = f.lookup(test_ds)

    input_shape = feats[0].shape[1]
    model = SimpleMLP(input_shape,100,3)

    if gpu():
        model.cuda()

    model.load_state_dict(torch.load("models/{0}.model".format(mname)))
    # print_evaluation(model, feats, FEVERLabelSchema(),args.log)
    #remove args.log
    label = print_evaluation(model, feats, FEVERLabelSchema())

    js = None
    with jsonlines.open("{0}.pages.p{1}.jsonl".format(split,k),"r") as f1:
        with jsonlines.open("../{0}.pages.p5.jsonl".format(split),"w") as f2:
            for line in f1:
                line["label"] = label
                f2.write(line)
