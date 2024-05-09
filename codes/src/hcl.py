## hcl -- Gaspard Berthelier
#utilities for project with HCL

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification 
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns
from datasets import Dataset
import utils

def get_tokenizer(name):
    print("Current dir : ",os.getcwd())
    path = f"models/{name}"
    print(f"Loading from path : {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    return tokenizer

def acc_fct(labels,prediction):
    """local accuracy function"""
    idx_true = np.where(np.array(labels)==np.array(prediction))[0]
    return len(idx_true)/len(labels)


def show_df(df,show_labels=True,label_names=True):
    """plots the df distributions"""
    fig,axes = plt.subplots(1,3,figsize=(12,3))
    sns.histplot(ax=axes[0],x=df["length"]).set(title="Number of characters")
    sns.histplot(ax=axes[1],x=df["words"]).set(title="Number of words")
    sns.histplot(ax=axes[2],x=df["unique_words"]).set(title="Number of unique words")
    fig.tight_layout()
    plt.show()
    mean_length = round(np.mean(df["length"].values),1)
    mean_words = round(np.mean(df["words"].values),1)
    mean_uniques = round(np.mean(df["unique_words"].values),1)

    print("Mean length: ",mean_length)
    print("Mean number of words: ",mean_words)
    print("Mean number of unique words: ",mean_uniques)
    
    if show_labels:
        labels = np.unique(df["label"])
        fig = plt.figure(figsize=(8,3))
        if label_names:
            label_count = {key:len(np.where(df["label"]==key)[0]) for key in labels}
            sns.barplot(x=labels,y=[label_count[key] for key  in labels])
            plt.tick_params(axis="x",rotation=90)
        else:
            sns.histplot(df["label"]).set(title="Label distribution")
        plt.show()
        return labels, (mean_words,mean_uniques,mean_length)
    
def get_label_ds(df,tokenizer,seed):
    """returns dataset from input dataframe"""
    dataset = Dataset.from_dict({"text":df["text"].values,"label":df["label"].values})
    dataset = utils.tokenize_dataset(dataset,tokenizer,is_hcl=True)
    dataset = dataset.shuffle(seed=seed)
    reduce_size = 850 #to have same dataset shape as BLUE experiment
    if df.shape[0]>reduce_size:
        dataset = dataset.select(range(reduce_size))
    return dataset

def truncate(dataset,max_size):
    """truncates texts to max_siz (character length)"""
    dataset = dataset.rename_column("text","long_text")
    dataset = dataset.map(lambda data: {"text":data["long_text"][0:max_size]})
    dataset = dataset.remove_columns(["long_text"])
    return dataset

