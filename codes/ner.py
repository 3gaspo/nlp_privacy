## ner - Gaspard Berthelier
# utilities for B/B' model


#imports
import json
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer,pipeline


#dataset
ner_path = "models/model_B"

def get_ner_df():
    """returns dataframe which was used to train NER model"""
    with open("data/anonymisation-ner.jsonl") as file:
        data = file.readlines()
    ds = {"text":[],"labels":[]}
    for entry in data:
        dico = json.loads(entry)
        ds["text"].append(" ".join(dico["tokens"]))
        ds["labels"].append(dico["ner_tags"])
    df = pd.DataFrame.from_dict(ds)
    df["words"] = df.text.apply(lambda data: len(data.split(" ")))
    df["unique_words"] = df.text.apply(lambda data: len(np.unique(data.split(" "))))
    df["length"] = df.text.apply(lambda data: len(data))
    df["private_tokens"] = df.labels.apply(lambda data: len(np.where(np.array(data)!=0)[0]))
    return df

def get_ner_tokenizer():
    """returns tokenizer for NER model"""
    tokenizer = AutoTokenizer.from_pretrained(ner_path)
    return tokenizer


#model
def get_ner_generator(tokenizer):
    """returns NER generator"""
    generator = pipeline(task="ner",model=ner_path,tokenizer=tokenizer)
    #the index returned are of extended words (get_words)
    return generator


#processing
def get_words(tokenizer,sequence):
    """returns the different string tokens"""
    encoded = tokenizer(sequence,truncation=True)["input_ids"]
    decoded = [tokenizer.decode(token) for token in encoded] #extended words format
    return decoded

def add_spaces(sentence):
    """replaces bugged spaces"""
    new_sentence=""
    for k in range(len(sentence)):
        if sentence[k] in ["_","▁"]:
            new_sentence+=" "
        else:
            new_sentence+= sentence[k]
    return new_sentence

def get_idx(output):
    """returns private phrases and corresponding index tokens"""
    private_phrases = [] #consecutive private tokens
    private_idx = [] #their index in extended words format
    temp_phrase = ""
    temp_idx = []
    for i,entity in enumerate(output):
        if i>0 and entity["entity"][0]=="B":
            private_phrases.append(add_spaces(temp_phrase))
            temp_phrase = entity["word"]
            private_idx.append(temp_idx)
            temp_idx = [entity["index"]]
        else:
            temp_phrase += entity["word"]
            temp_idx.append(entity["index"])
    private_phrases.append(add_spaces(temp_phrase))
    private_idx.append(temp_idx)
    return private_idx,private_phrases

def split_words(L):
    """return separate private words"""
    new_L = []
    for words in L:
        new_L += words.split(" ")
    return new_L

def print_private(L,words=None,index=None,join=True):
    prints = []
    if words:
        for word in L:
            if word in words:
                prints.append(f"\x1b[31m{word}\x1b[0m")
            else:
                prints.append(word)
    if index: #marche moins bien car tokens séparés par un espace (dans le même mot)
        for i,word in enumerate(L):
            if i in index:
                prints.append(f"\x1b[31m{word}\x1b[0m")
            else:
                prints.append(word)
    if join:
        print(" ".join(prints))
    else:
        print(prints)
        
        




## ner -- Gaspard Berthelier
# utilities for ner models

def true_count(predictions,labels):
    """returns counts of true values for private and not private terms"""
    counts = [0,0]
    for i,label in enumerate(labels):
        if label==1:#private
            if label==predictions[i]:
                counts[1]+=1
        else:
            if label==predictions[i]:
                counts[0]+=1
    return counts

def ner_acc(predictions_list,labels_list):
    """returns accuracy for private and not private"""
    total_counts = [0,0]
    totals = [0,0]
    for i,predictions in enumerate(predictions_list):
        counts = true_count(predictions,labels_list[i])
        for k in [0,1]:
            total_counts[k]+=counts[k]
            totals[k]+=len(np.where(np.array(labels_list[i])==k))

    accs = [None,None]
    for k in [0,1]:
        accs[k]=total_counts[k]/totals[k]
    
    return accs


def get_label_predictions(generator,sentence):
    """returns predictions for each word in a sentence"""
    words = sentence.split(" ")
    predicted_private_words = []
    predicted_private_score = []
    entities = generator(sentence)
    for entity in entities:
        predicted_private_words.append(entity["word"])
        predicted_private_score.append(entity["score"])
    
    return None


def compare_perf(model,model0,dataset):
    """compares NER perfomance of model vs baseline"""
    predictions = [get_label_predictions(model,sentence) for sentence in dataset["text"]]
    predictions0 = [get_label_predictions(model0,sentence) for sentence in dataset["text"]]
    accs = ner_acc(predictions,dataset["labels"])
    accs0 = ner_acc(predictions,dataset["labels"])
    return np.array(accs)-np.array(accs0)




#dump

'''
#utile pour accuracy de labelling probablement
def extend_label(tokenizer,text,labels):
    """returns labels of each tokens"""
    words = text.split(" ")
    extended_words = get_words(tokenizer,text)
    extended_labels = [0]
    j=1
    for i,word in enumerate(words):
        step=0
        label = labels[i]
        while step==0:
            temp_subword = extended_words[j]
            subword=add_spaces(temp_subword)
            if subword in word:
                extended_labels.append(label)
                j+=1
            else:
                step=1
    extended_labels.append(0)
    return extended_labels


def un_nest(L):
    new_L = []
    for words in L:
        new_L += words
    return new_L
    






def get_private_labels(output,long_text):
    idx = un_nest(get_idx(output))
    L = []
    n=len(long_text)
    for k in rang(n):
        if k in idx:
            L.append(1)
        else:
            L.append(0)
    return L  
'''