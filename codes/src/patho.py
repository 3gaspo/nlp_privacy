## patho - Gaspard Berthelier
# utilities for pathology dataset

## imports

#technical
from transformers import AutoModelForSequenceClassification

#gaspard
import hcl
import mem
import mia

#usuals
import pandas as pd
import numpy as np


## dataset
def get_label_df(seed,num_labels=10):
    """retuens pathology dataset"""
    train_path = "data/anonymized-data/cr-db-for-classif-Patho-Components-2021-11-11-03--TRAIN-anon.xlsx"
    df = pd.read_excel(train_path)
    df.drop(columns=["LABEL","index","Unnamed: 0.1","Unnamed: 0.3","Unnamed: 0.2","Unnamed: 0","Unnamed: 0.1.1","ANON_ID","UNIQUEID","PAT_ID","Unnamed: 10","Unnamed: 11"],axis=1,inplace=True)
    labels = np.unique(df["LABEL_TXT"])
    df["label"] = df.LABEL_TXT.apply(lambda x: np.argwhere(labels==x)[0][0])
    df.rename(columns={"TEXT":"text","LABEL_TXT":"label_txt"},inplace=True)
    df["words"] = df.text.apply(lambda x: len(x.split(" ")))
    df["unique_words"] = df.text.apply(lambda x: len(np.unique(x.split(" "))[0]))
    df["length"] = df.text.apply(lambda x: len(x))
    df = df.iloc[np.where(np.array(df["label"].values)<10)]
    labels = labels[0:num_labels]
    return df.sample(frac=0.1,random_state=seed).reset_index(drop=True), labels

def get_all_patho(real_t=0.2,num_labels=10,seed=42,print_graphs=True,compute_mem=False):
    
    df,labels = get_label_df(seed)
    if print_graphs:
        _, means = hcl.show_df(df)
        print("Size : ",df.shape)
        print("Labels : ",labels)

    tokenizer = hcl.get_tokenizer("bert")
    dataset = hcl.get_label_ds(df,tokenizer,seed)
    label_dicts = [{k:labels[k] for k in range(len(labels))},{key:k for k,key in enumerate(labels)}]
    
    real_mems = mem.mem_pipeline(dataset,tokenizer,label_dicts,means,real_t=real_t,n_counters=10,do_train=compute_mem,do_perf=compute_mem,model_dir="mem",seed=seed,num_train_epochs=3,print_results=True,is_hcl=True)
    dataset = mia.get_in_out(dataset, real_mems, seed=seed,out_size=0.5)
    return dataset, tokenizer, real_mems, label_dicts


## model
def get_model(name,label_dicts):
    path = f"models/{name}"
    id2label, label2id, num_labels = label_dicts[0], label_dicts[1], len(list(label_dicts[0].values()))
    return AutoModelForSequenceClassification.from_pretrained(path,num_labels=num_labels,id2label=id2label,label2id=label2id)