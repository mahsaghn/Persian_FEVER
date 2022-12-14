#!/usr/bin/env python3

#Adapted from https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A script to build the tf-idf document matrices for retrieval."""
import os
import pathlib
import torch
from common.util.log_helper import LogHelper
from drqascripts.retriever.build_tfidf import *


if __name__ == '__main__': 
    LogHelper.setup()
    logger = LogHelper.get_logger("DrQA Build TFIDF")
    LogHelper.get_logger("DRQA")

    logger.info("Build TF-IDF matrix")


    #step 1
    parser = argparse.ArgumentParser()
    
    #define db path
    parser.add_argument('db_path', type=str, default=None,
                        help='Path to sqlite db holding document texts')
    
    
    parser.add_argument('out_dir', type=str, default=None,
                        help='Directory for saving output files')
    
    #set ngram for tfidf --> 2  
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    
    #hash size is 16777216
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    
    #use tokenizer method in this case it is simple tokenizer
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))

    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    
    #return data from the option epecified
    args = parser.parse_args()

    #step2
    #in init file retriever.get_class is used
    #it gets data ready for analysiinit("heloooo")s.
    #downloading, cleaning, and standardizing datasets, and importing them into relational databases, flat files, or programming languages.    
    tfidf_builder = TfIdfBuilder(args,'sqlite', {'db_path': args.db_path})
    
    logging.info('Counting words...')
    count_matrix, doc_dict = tfidf_builder.get_count_matrix()

    logger.info('Making tfidf vectors...')
    tfidf_mat = tfidf_builder.get_tfidf_matrix(count_matrix)

    logger.info('Getting word-doc frequencies...')
    freqs = tfidf_builder.get_doc_freqs(count_matrix)

    #step3
    basename = os.path.splitext(os.path.basename(args.db_path))[0]
    basename += ('-tfidf-ngram=%d-hash=%d-tokenizer=%s' %
                 (args.ngram, args.hash_size, args.tokenizer))
    filename = os.path.join(args.out_dir, basename)

    logger.info('Saving to %s.npz' % filename)
    metadata = {
        'doc_freqs': freqs,
        'tokenizer': args.tokenizer,
        'hash_size': args.hash_size,
        'ngram': args.ngram,
        'doc_dict': doc_dict
    }
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True) 
    #save file
    retriever.utils.save_sparse_csr(filename, tfidf_mat, metadata)
    