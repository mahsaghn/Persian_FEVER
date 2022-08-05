import argparse

import torch
import os

from common.dataset.data_set import DataSet
from common.dataset.reader import JSONLineReader
from common.features.feature_function import Features
from common.training.options import gpu
from common.training.run import  print_evaluation
from common.util.log_helper import LogHelper
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
    #step1
    LogHelper.setup()
    logger = LogHelper.get_logger(__name__)

    #step2
    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str, help='db file path')
    parser.add_argument('test', type=str, help='test file path')
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument("--sentence", type=str2bool, default=False)
    parser.add_argument("--log",type=str,default=None)
    args = parser.parse_args()

    #step3
    logger.info("Loading DB {0}".format(args.db))
    db = FeverDocDB(args.db)

    #step4
    mname = args.model
    logger.info("Model name is {0}".format(mname))


    #step5
    print('step5')
    ffns = []

    if args.sentence:
        logger.info("Model is Sentence level")
        ffns.append(SentenceLevelTermFrequencyFeatureFunction(db, naming=mname))
    else:
        logger.info("Model is Document level")
        ffns.append(TermFrequencyFeatureFunction(db,naming=mname))

    #step6
    f = Features(mname,ffns)
    #step7
    print('step6',mname)
    f.load_vocab(mname)

    #step8
    print('step8')
    jlr = JSONLineReader()
    #step9
    formatter = FEVERGoldFormatter(None, FEVERLabelSchema())

    #step10
    test_ds = DataSet(file=args.test, reader=jlr, formatter=formatter)
    #step11
    test_ds.read()
    # print("____",test_ds.data)
    #step12
    feats = f.lookup(test_ds)
    #step13
    input_shape = feats[0].shape[1]
    model = SimpleMLP(input_shape,100,3)
    #step14
    if gpu():
        model.cuda()
    #step15
    model.load_state_dict(torch.load("models/{0}.model".format(mname)))
    print_evaluation(model, feats, FEVERLabelSchema(),args.log)
    #remove args.log
    #step16
    # print_evaluation(model, feats, FEVERLabelSchema())
