## hoc - Gaspard Berthelier
# for hoc dataset utilities

## imports
#technical
from datasets import Dataset

#gaspard
import utils
import mem
import mia

#usuals
import os
import re
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

#file extraction
def get_text(file_path):
    """returns a string text from file"""
    with open(file_path,encoding="utf8") as text_file:
        texts = text_file.readlines()
    return texts

def remove_str(text,pattern):
    """removes a redundant string pattern in a list of texts"""
    removing = True
    while removing:
        try:
            text.remove(pattern)
        except:
            removing=False
    return text

def get_label(file_path):    
    """returns a string label from file"""    
    with open(file_path,encoding="utf8") as label_file:
        label = label_file.readlines()[0]

    label = label.split("<")
    for pattern in [" ",""]:
        label = remove_str(label,pattern)
    label = " ".join(label)
    
    label = label.split(" ")
    for pattern in [" ",""]:
        label = remove_str(label,pattern)
    label = " ".join(label)
 
    return label
    
def get_hoc():
    """returns hoc dataset as text and labels strings"""
    texts = []
    labels = []
    os.chdir("text")
    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{file}"
            texts.append(get_text(file_path))

    os.chdir("../labels")
    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{file}"
            labels.append(get_label(file_path))
            
    os.chdir("..")
    
    return texts,labels
        
def label_cleaning(label):
    """seperates mutli labels from label string"""
    delimiters = "--","AND","and","  "
    regex_pattern = '|'.join(map(re.escape, delimiters))
    labels = re.split(regex_pattern, label)
    for i,label in enumerate(labels):
        labels[i] = label.strip()
    remove_str(labels," ")
    remove_str(labels,"")
    output = list(dict.fromkeys(labels))
    return output if output != [] else None


hallmarks = [
    "Sustaining proliferative signaling",
    "Evading growth suppressors",
    "Resisting cell death",
    "Enabling replicative immortality",
    "Inducing angiogenesis",
    "Activating invasion & metastasis",
    "Genome instability & mutation",
    "Tumor-promoting inflammation",
    "Deregulating cellular energetics",
    "Avoiding immune destruction"]

hallmarks2 = [
    "proliferative signaling",
    "growth suppressors",
    "cell death",
    "replicative immortality",
    "angiogenesis",
    "invasion",
    "instability",
    "tumor-promoting",
    "deregulating",
    "avoiding"]

hallmarks3 = [
    "proliferative",
    "suppressors",
    "cell death",
    "immortality",
    "angiogenesis",
    "metastasis",
    "mutation",
    "inflammation",
    "energetics",
    "destruction"]

def get_hallmarks():
    """returns hallmark names"""
    return hallmarks

#dataset

def detect_label(labels):
    """returns one hot vector of label presence"""
    label_presence = [0 for k in range(len(hallmarks))]
    for label in labels:
        for hallmark_dict in [hallmarks,hallmarks2,hallmarks3]:
            for i,hallmark in enumerate(hallmark_dict):
                if hallmark in label:
                    label_presence[i]=1
    return label_presence


def get_mono_dfds(tokenizer,seed,print_info=True):
    """returns cleaned hoc dataset"""
    print("Building dataset")
    texts,labels = get_hoc()
    df = pd.DataFrame({"text":texts,"labels":labels})
    df["labels"] = df["labels"].apply(lambda x: label_cleaning(x)) #list of strings
    df["label_presence"] = df["labels"].apply(lambda x: detect_label(x) if x else None) #one hot vector
    df["label_counts"] = df["label_presence"].apply(lambda x: np.sum(x) if x else None) #int
    df.drop(np.where(df["label_counts"]==0)[0],axis=0,inplace=True) #remove lines without labels
    df.reset_index(drop=True, inplace=True)
    df.drop(np.where(df["label_presence"].values==None)[0],axis=0,inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(np.where(df["label_counts"].values==None)[0],axis=0,inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["text"] = df["text"].apply(lambda x: " ".join(x))
    df["word_count"] = df["text"].apply(lambda x: len(x.split(" ")))
    df["unique_words"] = df["text"].apply(lambda x: len(np.unique(x.split(" "))))
    df["chr_count"] = df["text"].apply(lambda x: len(x))
    N = df.shape[0]
    df[hallmarks] = [df["label_presence"].values[k] for k in range(N)] #one hot column for each label
    mono_idx = np.where(df["label_counts"]==1)[0]
    df_mono = df.iloc[mono_idx]
    
    dataset = Dataset.from_dict({"text":df["text"].values,"label":df["label_presence"].values}) #labels are one-hot list
    dataset = utils.tokenize_dataset(dataset,tokenizer)
    dataset_mono = dataset.select(mono_idx)
    dataset_mono = dataset_mono.map(lambda data: {"label":np.argmax(data["label"])}) #ordinal label
    dataset_mono = dataset_mono.shuffle(seed=seed)
    
    if print_info:
        utils.check_dataset(dataset_mono)
        long_idx = np.where(np.array(utils.get_num_tokens(dataset_mono["input_ids"]))==512)[0]
        print(f"Truncated {len(long_idx)} sequences")
    
    return dataset_mono, df_mono

def plot_mono_dfds(dataset_mono,df_mono,hallmarks):
    """plots dataset info"""
    fig, axes = plt.subplots(2, 3,figsize=(15, 7))
    sns.histplot(ax = axes[0,0],x=df_mono["word_count"]).set(title="Number of words distribution")
    sns.histplot(ax = axes[0,1],x=df_mono["unique_words"]).set(title="Number of unique words distribution")
    sns.histplot(ax = axes[1,0],x=utils.get_num_tokens(dataset_mono["input_ids"])).set(title="Number of tokens distribution")
    sns.histplot(ax = axes[1,1],x=utils.get_unique_tokens(dataset_mono["input_ids"])).set(title="Number of unique tokens distribution")
    sns.histplot(ax = axes[0,2],x=df_mono["chr_count"]).set(title="Number of characters distribution")
    label_counts = {key:sum(df_mono[key].values) for key in hallmarks}
    sns.barplot(ax = axes[1,2],x = [str(k) for k in range(len(hallmarks))], y = [label_counts[key] for key in hallmarks]).set(title="Labels distribution")
    plt.tick_params(axis='x', rotation=90)
    fig.tight_layout()

def get_hoc_ds(tokenizer,seed,print_graphs=True,return_means=True):
    """returns hoc dataset"""
    dataset_mono, df_mono = get_mono_dfds(tokenizer,seed,print_info=print_graphs)
    if print_graphs:
        plot_mono_dfds(dataset_mono,df_mono,get_hallmarks())
    mean_words = int(np.mean(df_mono["word_count"]))
    mean_uniques = int(np.mean(df_mono["unique_words"]))
    mean_size = int(np.mean(df_mono["chr_count"]))
    if print_graphs:
        print("Mean number of word : ",mean_words)
        print("Mean number of unique words : ",mean_uniques)
        print("Mean number of characters : ",mean_size)
    if return_means:
        return dataset_mono, mean_words, mean_uniques, mean_size
    else:
        return dataset_mono
    
    
def get_all_hoc(seed=42,real_t=0.3,print_graphs=True,model_name="bert",include_mem=True,compute_mem=False,mem_dir="mem"):
    """returns dataset memorization and other info"""
    tokenizer = utils.get_tokenizer(model_name)
    dataset, mean_words, mean_uniques, mean_sizes = get_hoc_ds(tokenizer,seed,print_graphs=print_graphs)
    means = (mean_words, mean_uniques, mean_sizes)
    id2label, label2id = utils.get_label_dict(10)
    label_dicts = [id2label,label2id]
    real_mems = mem.mem_pipeline(dataset,tokenizer,label_dicts,means,real_t=real_t,n_counters=10,do_train=compute_mem,do_perf=compute_mem,model_dir=mem_dir,seed=seed,num_train_epochs=3,print_results=print_graphs)
    dataset = mia.get_in_out(dataset,real_mems,seed=seed,out_size=0.5,include_mem=include_mem)
    return dataset, real_mems, tokenizer, label_dicts
