import json

import torch
import torch.nn.functional as F
from sklearn.utils import shuffle

from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from common.training.batcher import Batcher, prepare, prepare_with_labels
from common.util.random import SimpleRandom


def evaluate(model,data,labels,batch_size):
    predicted = predict(model,data,batch_size)
    return accuracy_score(labels,predicted.data.numpy().reshape(-1))

def predict(model, data, batch_size):
    batcher = Batcher(data, batch_size)
    # print("1.......",batcher)
    predicted = []
    for batch, size, start, end in batcher:
        # print('2...batch{0}...size{1}...start{2}....end{3}'.format(batch,size,start,end))
        d = prepare(batch)
        # print("3....",d)
        model.eval()
        #The model returns a Dict[Tensor] during training, 
        # containing the classification and regression losses and the mask loss.
        logits = model(d).cpu()
        # print("4....",)
        # print("00___________00________",torch.max(logits, 1))
        predicted.extend(torch.max(logits, 1)[1])
    return torch.stack(predicted)

def train(model, fs, batch_size, lr, epochs,dev=None, clip=None, early_stopping=None,name=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    data, labels = fs
    print("____",data.shape)
    if dev is not None:
        dev_data,dev_labels = dev

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_data = 0

        shuffle(data,labels)

        batcher = Batcher(data, batch_size)

        for batch, size, start, end in batcher:
            d,gold = prepare_with_labels(batch,labels[start:end])

            model.train()
            optimizer.zero_grad()
            logits = model(d)
            # print("logits..",logits)
            # print("gold....",gold)
            loss = F.cross_entropy(logits, gold)
            loss.backward()

            epoch_loss += loss.cpu()
            epoch_data += size

            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()

        print("Average epoch loss: {0}".format((epoch_loss/epoch_data).data.numpy()))

        #print("Epoch Train Accuracy {0}".format(evaluate(model, data, labels, batch_size)))
        if dev is not None:
            acc = evaluate(model,dev_data,dev_labels,batch_size)
            print("Epoch Dev Accuracy {0}".format(acc))

            if early_stopping is not None and early_stopping(model,acc):
                break

    if dev is not None and early_stopping is not None:
        early_stopping.set_best_state(model)



def print_evaluation(model,data,ls,log=None):
    features,actual = data
    predictions = predict(model, features, 1000).data.numpy().reshape(-1).tolist()
    labels = [ls.idx[i] for i, _ in enumerate(ls.idx)]
    actual = [labels[i] for i in actual]
    predictions = [labels[i] for i in predictions]
    bold = '\033[1m'
    end = '\033[0m'
    print("____________________________________________________",len(actual),len(predictions))
    print(bold+'accuracy_score:\n '+end,accuracy_score(actual, predictions))
    print("____________________________________________________")
    print(bold+'classification_report:\n\n '+end,classification_report(actual, predictions))
    print("____________________________________________________")
    print(bold+'confusion_matrix:'+end+'\nCompute confusion matrix to evaluate the accuracy of a classification.\nnumber of classes-->n*n matrix\nclass0:supports\nclass1:refutes\nclass3:NotEnoughInfo\n\n'
    ,confusion_matrix(actual, predictions))

    data = zip(actual,predictions)
    # print("______",actual[:10],predictions[:10])
    if log is not None:
        #this code is changed due to logs
        
        f = open(log, "w+")
        for a,p in data:
            f.write(json.dumps({"actual": a, "predicted": p}) + "\n")
        f.close()
    return predictions
