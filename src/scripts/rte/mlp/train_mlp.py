# import nltk
# nltk.download('punkt')

import argparse

import torch
import os

from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from common.features.feature_function import Features
from common.training.early_stopping import EarlyStopping
from common.training.options import gpu
from common.training.run import train, print_evaluation
from common.util.log_helper import LogHelper
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
from rte.riedel.fever_features import TermFrequencyFeatureFunction
from rte.riedel.model import SimpleMLP
from rte.riedel.sent_features import SentenceLevelTermFrequencyFeatureFunction


def model_exists(mname):
    return os.path.exists(os.path.join("models","{0}.model".format(mname)))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    SimpleRandom.set_seeds()

    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)
    # kgropgk
    #step1
    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str, help='db file path')
    parser.add_argument('train', type=str, help='train file path')
    parser.add_argument('dev', type=str, help='dev file path')
    parser.add_argument('--test', required=False ,type=str, default=None ,help="test file path")
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument("--sentence",type=str2bool, default=False)
    parser.add_argument("--filtering",type=str, default=None)
    args = parser.parse_args()
    #step2
    if not os.path.exists("models"):
        os.mkdir("models")

    #step3
    logger.info("Loading DB {0}".format(args.db))
    db = FeverDocDB(args.db)
    #step4
    mname = args.model
    logger.info("Model name is {0}".format(mname))

    #step5
    ffns = []

    if args.sentence:
        logger.info("Model is Sentence level")
        print("___args.sentence")
        ffns.append(SentenceLevelTermFrequencyFeatureFunction(db, naming=mname))
    else:
        print("___args.sentence")
        logger.info("Model is Document level")
        ffns.append(TermFrequencyFeatureFunction(db,naming=mname))

    #step6
    f = Features(mname,ffns)
    #step7
    jlr = JSONLineReader()

    #step8
    formatter = FEVERGoldFormatter(None, FEVERLabelSchema(),filtering=args.filtering)

    #step9
    train_ds = DataSet(file=args.train, reader=jlr, formatter=formatter)
    #step10
    dev_ds = DataSet(file=args.dev, reader=jlr, formatter=formatter)

    #step11
    train_ds.read()
    #step12
    dev_ds.read()
    #step13
    test_ds = None
    if args.test is not None:
        test_ds = DataSet(file=args.test, reader=jlr, formatter=formatter)
        test_ds.read()
    #step14
    print("+++",len(test_ds.data))
    train_feats, dev_feats, test_feats = f.load(train_ds, dev_ds,test_ds)
    print("+++++",train_feats[0].shape,dev_feats[0].shape,test_feats[0].shape)
    #step15
    input_shape = train_feats[0].shape[1]
    model = SimpleMLP(input_shape,100,3)

    #step16
    if gpu():
        model.cuda()

    #step17
    if model_exists(mname) and os.getenv("TRAIN","").lower() not in ["y","1","t","yes"]:
        model.load_state_dict(torch.load("models/{0}.model".format(mname)))
    else:
        train(model, train_feats, 50, 1e-2, 90,dev_feats,early_stopping=EarlyStopping(mname))
        torch.save(model.state_dict(), "models/{0}.model".format(mname))


    #step18
    print_evaluation(model, dev_feats, FEVERLabelSchema())

    #step19
    if args.test is not None:
        print_evaluation(model, test_feats, FEVERLabelSchema())