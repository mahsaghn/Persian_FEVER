import argparse
import jsonlines
import torch
import os
import json

from random import seed
from random import choice
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import random
seed(15)
data = []
labels = []
srcpages = []
numne=0
nums=0
numr=0
refi =[]
supi = []
neii = []
train = data
with jsonlines.open("data/fever-data/dataset.jsonl","r") as f1:
    for i,line in enumerate(f1):
        # js = json.loads(line)
        data.append(line)
        labels.append(line['label'])
        srcpages.append(line['claim_page_token'])
        if line['label'] == 'REFUTES':
            numr+=1
            refi.append(i)
        elif line['label'] == 'SUPPORTS':
            nums+=1
            supi.append(i)
        else:
            numne +=1
            neii.append(i)

print("______________",len(data))
print("SUPPORTS______________",nums)
print("REFUTES______________",numr)
print("notenough______________",numne)
minimalgp = min(numne,numr,nums)
numtest = int(minimalgp*0.1)
print('numtest',numtest)
data = np.array(data)
labels = np.array(labels)
srcpages = np.array(srcpages)
numne=0
nums=0
numr=0
test = []
while( nums<numtest or numr<numtest or numne<numtest):
    minlen =  min(numne,numr,nums)
    if minlen == numr:
        while 1:
            index = choice(refi)
            page = data[index]['claim_page_token']
            lne =len([d for d in data if d['claim_page_token']==page and d['label'] == "NOT ENOUGH INFO"])
            ls =len([d for d in data if d['claim_page_token']==page and d['label'] == "SUPPORTS"])
            lr =len([d for d in data if d['claim_page_token']==page and d['label'] == "REFUTES"])
            # print('R:le',lne,"---ls",ls,'----lr',lr)
            if lr*1.2>=lne and lr*1.2>=ls:
                break
    elif minlen == numne:
        while 1:
            index = choice(neii)
            page = data[index]['claim_page_token']
            lne =len([d for d in data if d['claim_page_token']==page and d['label'] == "NOT ENOUGH INFO"])
            ls =len([d for d in data if d['claim_page_token']==page and d['label'] == "SUPPORTS"])
            lr =len([d for d in data if d['claim_page_token']==page and d['label'] == "REFUTES"])
            # print('NE:le',lne,"---ls",ls,'----lr',lr)
            if lne*1.2>=lr and lne*1.2>=ls:
                break
    else:
        while 1:
            index = choice(supi)
            page = data[index]['claim_page_token']
            lne =len([d for d in data if d['claim_page_token']==page and d['label'] == "NOT ENOUGH INFO"])
            ls =len([d for d in data if d['claim_page_token']==page and d['label'] == "SUPPORTS"])
            lr =len([d for d in data if d['claim_page_token']==page and d['label'] == "REFUTES"])
            # print('S:le',lne,"---ls",ls,'----lr',lr)
            if ls*1.2>=lne and ls*1.2>=lr:
                break

    page = data[index]['claim_page_token']
    ne =[i for i,d in enumerate(data) if d['claim_page_token']==page and d['label'] == "NOT ENOUGH INFO"]
    s =[i for i,d in enumerate(data) if d['claim_page_token']==page and d['label'] == "SUPPORTS"]
    r =[i for i,d in enumerate(data) if d['claim_page_token']==page and d['label'] == "REFUTES"]
    lne = len(ne)
    ls = len(s)
    lr = len(r)
    if nums+ls>1.2*numtest or numr+lr>1.2*numtest or numne+lne>1.2*numtest :
        continue
    refi = [j for j in refi if j not in r]
    neii = [j for j in neii if j not in ne]
    supi = [j for j in supi if j not in s]
    numne += lne
    numr += lr
    nums += ls
    test.extend([d for d in data if d['claim_page_token']==page])
    train = [d for d in train if d['claim_page_token']!=page]
print('test:  nei',numne,"---nums",nums,'----numr',numr)


numne=0
nums=0
numr=0
val = []
seed(12)
while( nums<numtest or numr<numtest or numne<numtest):
    minlen =  min(numne,numr,nums)
    if minlen == numr:
        while 1:
            index = choice(refi)
            page = data[index]['claim_page_token']
            lne =len([d for d in data if d['claim_page_token']==page and d['label'] == "NOT ENOUGH INFO"])
            ls =len([d for d in data if d['claim_page_token']==page and d['label'] == "SUPPORTS"])
            lr =len([d for d in data if d['claim_page_token']==page and d['label'] == "REFUTES"])
            if lr*1.5>=lne and lr*1.3>=ls:
                break
    elif minlen == numne:
        while 1:
            index = choice(neii)
            page = data[index]['claim_page_token']
            lne =len([d for d in data if d['claim_page_token']==page and d['label'] == "NOT ENOUGH INFO"])
            ls =len([d for d in data if d['claim_page_token']==page and d['label'] == "SUPPORTS"])
            lr =len([d for d in data if d['claim_page_token']==page and d['label'] == "REFUTES"])
            if lne*1.5>=lr and lne*1.3  >=ls:
                break
    else:
        while 1:
            index = choice(supi)
            page = data[index]['claim_page_token']
            lne =len([d for d in data if d['claim_page_token']==page and d['label'] == "NOT ENOUGH INFO"])
            ls =len([d for d in data if d['claim_page_token']==page and d['label'] == "SUPPORTS"])
            lr =len([d for d in data if d['claim_page_token']==page and d['label'] == "REFUTES"])
            if ls*1.3>=lne and ls*1.3>=lr:
                break

    page = data[index]['claim_page_token']
    ne =[i for i,d in enumerate(data) if d['claim_page_token']==page and d['label'] == "NOT ENOUGH INFO"]
    s =[i for i,d in enumerate(data) if d['claim_page_token']==page and d['label'] == "SUPPORTS"]
    r =[i for i,d in enumerate(data) if d['claim_page_token']==page and d['label'] == "REFUTES"]
    lne = len(ne)
    ls = len(s)
    lr = len(r)
    if nums+ls>1.2*numtest or numr+lr>1.2*numtest or numne+lne>1.2*numtest :
        print(":(((")
        continue
    refi = [j for j in refi if j not in r]
    neii = [j for j in neii if j not in ne]
    supi = [j for j in supi if j not in s]
    numne += lne
    numr += lr
    nums += ls
    val.extend([d for d in data if d['claim_page_token']==page])
    train = [d for d in train if d['claim_page_token']!=page]
print('valid: nei',numne,"---nums",nums,'----numr',numr)
ne =len([i for d in train if d['label'] == "NOT ENOUGH INFO"])
s =len([i for d in train if  d['label'] == "SUPPORTS"])
r =len([i for d in train if d['label'] == "REFUTES"])
print('train: nei',ne,"---nums",s,'----numr',r)

print('train---',len(train))
print('val---',len(val))
print('test---',len(test))
train = np.array(train)
np.random.shuffle(train)
val = np.array(val)
np.random.shuffle(val)
test = np.array(test)
np.random.shuffle(test)
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=923/8923, random_state=42,shuffle=True)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/8, random_state=42,shuffle=True)
with jsonlines.open("data/fever-data/train.jsonl","w") as f2:
    # writer = jsonlines.Writer(f2)
    # writer.write_all(data[:4500])
    # writer.close()
    f2.write_all(train)

with jsonlines.open("data/fever-data/dev.jsonl","w") as f3:
    f3.write_all(val)

with jsonlines.open("data/fever-data/test.jsonl","w") as f4:
    f4.write_all(test)
