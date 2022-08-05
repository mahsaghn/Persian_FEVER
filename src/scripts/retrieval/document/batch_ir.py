import argparse
import json
from tqdm import tqdm
from drqa import retriever
from drqa.retriever import DocDB
import jsonlines

def process(ranker, query, k=1):
    print(query)
    doc_names, doc_scores = ranker.closest_docs(query, k)

    return zip(doc_names, doc_scores)



if __name__ == "__main__":
    #step1
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--split',type=str)
    parser.add_argument('--count',type=int, default=1)
    args = parser.parse_args()

    k = args.count
    split = args.split

    #step1.1
    ranker = retriever.get_class('tfidf')(tfidf_path=args.model)

    #step2
    with jsonlines.open("data/fever-data/{0}.jsonl".format(split),"r") as f:
        with jsonlines.open("data/fever/{0}.pages.p{1}.jsonl".format(split,k),"w") as f2:
            #step3
            for line in f:
                #a
                #b
                pages = process(ranker,line['claim'],k=k)
                #c
                # fkregkpop
                line["predicted_pages"] = list(pages)
                #d
                f2.write(line)






