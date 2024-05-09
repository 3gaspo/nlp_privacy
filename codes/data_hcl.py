# data_hcl -- Gaspard Berthelier
#utilities to load model A'

## imports
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForMaskedLM
import json
import pandas as pd
import numpy as np

## dataset
def get_data_df(data_type="train"):
    """returns dataframe for model A' (used to finetune camembert into medical)"""
    with open(f"data/data-for-trf-{data_type}.json",encoding="utf8") as file:
        data = json.load(file)
    
    ds = {"text":[],"type":[]}
    for entry in data:
        ds["text"].append(entry["file.contenu"])
        ds["type"].append(entry["type.libelle"])
    df = pd.DataFrame.from_dict(ds)
    df["words"] = df.text.apply(lambda data: len(data.split(" ")))
    df["unique_words"] = df.text.apply(lambda data: len(np.unique(data.split(" "))))
    df["length"] = df.text.apply(lambda data: len(data))
    labels = np.unique(df.type)
    df["label"]= df.type.apply(lambda x: np.argwhere(labels==x)[0][0])

    return df.iloc[np.where(df.length<10000)] #texts are too long

def get_tokenizer():
    """returns tokenizer for model A' """
    tokenizer_A_prime = AutoTokenizer.from_pretrained("models/model_A'/best")
    return tokenizer_A_prime


## model
def get_model(nom="model_A'/best"):
    """returns model A'"""
    A_prime = AutoModelForMaskedLM.from_pretrained(f"models/{nom}")
    return A_prime

def get_generator(tokenizer_A_prime):
    """returns generator for fillmasking with model A' """
    generator = pipeline(task="fill-mask",model="models/model_A'/best",tokenizer=tokenizer_A_prime)
    return generator