# Persian_FEVER
Fact Extraction and Verification in Persian

This project includes two main tasks; first to find the validity of a given claim.

Labels are:
- Supports
- Refutes
- Not Enough Info

The second is to find evidence for such a prediction. 

For a given claim, we search through the dataset for similar articles. P nearest articles are chosen. The next step is to find evidence from each article. In this step, top k sentences which have a higher TFiDF similarity rank are selected as evidence. 

## Model Architectures:
- MLP: Multi-Layer Perceptron
- RTE: Recognizing Textual Entailment



## Demo
A demo website is also available in this repository. 

## Others
This project is inspired by [fever](https://github.com/sheffieldnlp/naacl2018-fever) project. The dataset is replaced by a Persian dataset which is gathered by Mr M. Zarharan from [Wikipedia](https://fa.wikipedia.org/wiki/%D8%B5%D9%81%D8%AD%D9%87%D9%94_%D8%A7%D8%B5%D9%84%DB%8C) website articles. [Scripts folder](https://github.com/mahsaghn/Persian_FEVER/tree/main/scripts) contains bash scripts to download some of requierd data. 
