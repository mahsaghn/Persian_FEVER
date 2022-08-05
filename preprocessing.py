import os
import sys
sys.path = ['/F/University/7/fever/fever-persian', '/F/University/7/fever/fever-persian/src', '/usr/lib/python37.zip', '/usr/lib/python3.7', '/usr/lib/python3.7/lib-dynload', '/F/University/7/fever/fever/lib/python3.7/site-packages']
import string
import numpy as np
# from fact_extraction.FNCFeatureExtraction import FNCFeatureExtraction

import argparse

import torch
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
db = None
model = None

def tf_idf_sim(claim, lines,freqs=None):
    args = Namespace(tokenizer='simple' , hash_size= int(np.math.pow(2, 24)), ngram=2)
    tfidf = OnlineTfidfDocRanker(args,[line["sentence"] for line in lines],freqs)
    line_ids,scores = tfidf.closest_docs(claim,5)
    ret_lines = []
    for idx,line in enumerate(line_ids):
        ret_lines.append(lines[line])
        ret_lines[-1]["score"] = scores[idx]
    return ret_lines

def tf_idf_claim(line,db):
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
    predicted = predict(model, features, 1000).data.numpy().reshape(-1).tolist()
    print('++++++++++++++++++-',predicted,'__',actual)
    if predicted[0] == 0:
        return "SUPPORTS"
    elif predicted[0] == 1:
        return "REFUTES"
    return "NOT ENOUGH INFO"


def process(ranker, query, k=1):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    print("_____",doc_scores)
    return zip(doc_names, doc_scores)


def extract_fact(line):
    global model,db
    args = Namespace(db ='data/fever/fever.db' ,sentence = True,model='ns_nn_sent'
        ,modeltfidf='data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz',count=3,split="test")
    db = FeverDocDB(args.db)
    mname = args.model
    k = args.count
    split = args.split
    ranker = retriever.get_class('tfidf')(tfidf_path=args.modeltfidf)
    pages = process(ranker,line['claim'],k=k)
    line["predicted_pages"] = list(pages)
    line = tf_idf_claim(line,db)
    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)
    ffns = []

    if args.sentence:
        print("it is sentences")
        ffns.append(SentenceLevelTermFrequencyFeatureFunction(db, naming=mname))
    else:
        ffns.append(TermFrequencyFeatureFunction(db,naming=mname))


    f = Features(mname,ffns)
    f.load_vocab(mname)

    jlr = JSONLineReader()
    formatter = FEVERGoldFormatter(None, FEVERLabelSchema())
    
    with jsonlines.open("view.jsonl","w") as f32:
        f32.write(line)
    print("_____________________",line)
    test_ds = DataSet(file="view.jsonl", reader=jlr, formatter=formatter)
    test_ds.read()
    print("data is",test_ds.data)
    feats = f.lookup(test_ds)

    input_shape = feats[0].shape[1]
    model = SimpleMLP(input_shape,100,3)

    if gpu():
        model.cuda()

    model.load_state_dict(torch.load("models/{0}.model".format(mname)))
    label = print_evaluation(model, feats, FEVERLabelSchema())
    #remove args.log
    print("this is the predicted label",label)
    js = None
    line["ds_label"] = label
    return line

def findhtml(num_sent, page, sent, predicted_sents,class_name):
    for predicted_sent in predicted_sents:
        if num_sent == predicted_sent[1] and predicted_sent[0] == page:
            return '<span class={0}>{1}</span>'.format(class_name,sent)
    return '<span>{0}</span>'.format(sent)


def get_docs_html(pages,predicted_sent,class_name):
    docs = []
    hyperlinks = []
    page_titles = [p[0].replace("_"," ").replace('hyperlink','') for p in pages]
    for page in pages:
        html = ""
        page_lines = db.get_doc_lines(page[0])
        num_line = 0
        for page_line in page_lines.split('\n'):
            try:
                theline = page_line.split("\t")[1]
                thenum = int(page_line.split("\t")[0])
                if len(theline)>1:
                    html+= findhtml(thenum,page[0],theline,predicted_sent,class_name)
                else:
                    num_line = thenum
            except:
                if page_line != "" or page_line !=" ":
                    html+= findhtml(num_line,page[0],page_line,predicted_sent,class_name)
        html = html.replace("*"," ")
        page_title = page[0].replace("_"," ")
        if 'hyperlink' in page_title:
                page_title = page_title.replace('hyperlink','')
                # print("____",page_title,page_titles)
                if page_title not in page_titles:
                    hyperlinks.append((page_title+' (واژگان مورد نیاز)',html))
        else:
            docs.append((page_title,html))
    docs.extend(hyperlinks)
    return docs

def get_dataset_evidence(line):
    print("LINE IDDDDDDDDDDDDDDDDDD",line['id'])
    if line['id'] is not None:
        with jsonlines.open("data/fever-data/{0}.jsonl".format(line['file']),"r") as f2:
            for sample in f2:
                if sample['id'] == line['id'] :
                    print("##############################",sample)
                    if sample['label'] != "NOT ENOUGH INFO":
                        print("##############################")
                        evidence = []
                        for e in sample['evidence']:
                            for ee in e:
                                # print()
                                print("_____",ee)
                                evidence.append([ee[2],ee[3]])
                        line['ds_evidence'] = evidence
                        line['label'] = sample['label']
                        pages = []
                        for e in line['ds_evidence']:
                            if e[0] not in pages:
                                pages.append((e[0],0))
                        line['ds_pages']= pages
                        break
                    else:
                        line['ds_evidence'] = None
                        line['label'] = sample['label']
                        line['ds_pages']= None
                    break
        
    return line 
                    

# line = {"claim": "کارگردان فیلم مهتاب بری جنکینز متولد 19 نوامبر 1979 است."}
# extract_fact(line)
