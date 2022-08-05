# import os
# from keras.models import load_model
# from transformers import AutoConfig, AutoTokenizer, AutoModel
# from flair.embeddings import TransformerWordEmbeddings
# from hazm import *

# if os.path.exists('stance_detection/mymodel.h5'):
#     headline_model = load_model('stance_detection/mymodel.h5') 
#     print("headline model loaded")


# normalizer = Normalizer()
# init_stopWords = list()
# with open("stance_detection/stopWords.txt",'r' ) as f:
#     for line in f:
#         init_stopWords.append(normalizer.normalize(line.rstrip("\n\r")))
# print('__GetStopWords done!')

# parsbert_config = AutoConfig.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
# parsbert_tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
# parsbert_model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
# parsbert_tokenizer.model_max_length = 512
# embedding = TransformerWordEmbeddings('HooshvareLab/bert-base-parsbert-uncased', layers='-1,-2,-3,-4')
# embedding.tokenizer.model_max_length = 512
