## utils -- Gaspard Berthelier
# for basic utilities

#technical
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from evaluate import load
from transformers import Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error
import torch
from datasets import concatenate_datasets
metric = load("accuracy")

#gaspard
import hcl

#usuals
import numpy as np
from time import perf_counter
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

#debugging
import transformers
import datasets
import importlib
import os
def reload(name):
    importlib.reload(name)
    datasets.logging.set_verbosity_error()
    transformers.logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
#dataset
def get_tokenizer(text):
    """returns tokenizer for bert of distilbert"""
    name = text
    if text=="distilbert":
        name = "distilbert-base-uncased"
    elif text=="bert":
        name = "bert-base-uncased"
    return AutoTokenizer.from_pretrained(name)

def tokenizer_fct(example,tokenizer, is_hcl=False):
    """tokenizing function"""
    if is_hcl:
        return tokenizer(example["text"],truncation=True,padding='max_length',max_length=512)
    else:
        return tokenizer(example["text"],truncation=True,padding='max_length')
    
def tokenize_dataset(dataset,tokenizer,is_hcl=False):
    """tokenizes dataset"""
    return dataset.map(lambda data: tokenizer_fct(data,tokenizer,is_hcl=is_hcl),batched=True)

def sample_dataset(dataset,ratio,seed):
    """reduces size of dataset by ratio factor"""
    size = dataset.shape[0]
    return dataset.shuffle(seed=seed).select(range(int(ratio*size)))

def remove_cols(dataset):
    """returns datasets with only training columns"""
    cols = []
    for col in ["text","predicted_label","regression","prediction_score","true_label"]:
        if col in dataset.column_names:
            cols.append(col)
    return dataset.remove_columns(cols)
        

def check_dataset(dataset):
    """returns dataset info"""
    print("--------------")
    print("Num rows : ",dataset.num_rows)
    print("Columns : ",dataset.column_names)
    print("Unique labels : ",np.unique(dataset["label"]))
    
    dataset_col = remove_cols(dataset)
    print("Useful columns : ",dataset_col.column_names)
    dataset_test = [torch.tensor(dataset_col[key]) for key in dataset_col.column_names]
    print("Shapes : ",[dataset_test[k].shape for k in range(len(dataset_test))])
    
def get_label_dict(n):
    """returns label dictionaries"""
    id2label = {key:f"LABEL_{key}" for key in range(n)}
    label2id = {f"LABEL_{key}":key for key in range(n)}
    return id2label, label2id

def get_train_test(dataset,real_mems,seed=42,test_size=0.1):
    """returns in/out with mems inside in"""
    N = dataset.num_rows
    train_idx = np.unique([k for k in range(int(N*(1-test_size))) if k not in real_mems]+real_mems)
    in_ds = dataset.select(train_idx)
    out_ds = dataset.select([k for k in range(int(N*(1-test_size)),N) if k not in real_mems])
    return in_ds,out_ds

def get_word_count(L):
    """returns word count of sentences in list"""
    words = []
    for sent in L:
        words.append(len(sent.split(" ")))
    return words
def get_unique_count(L):
    """returns unique word count of sentences in list"""
    words = []
    for sent in L:
        words.append(len(np.unique(sent.split(" "))))
    return words
def get_size_count(L):
    """returns character count of sentences in list"""
    sizes = []
    for sent in L:
        sizes.append(len(sent))
    return sizes
def get_unique_tokens(L):
    """returns unique token count in list"""
    uniques = []
    for tokens in L:
        uniques.append(len(np.unique(tokens)))
    return uniques         
def get_num_tokens(L):
    """returns token count in list"""
    count = []
    for tokens in L:
        count.append(len(np.where(np.array(tokens)!=0)[0]))
    return count

def translate(label,true_label,score):
    """returns regression (label+score)"""
    if label==true_label:
        return label+score
    else:
        return -(label+score)

def separate(label,true_label,score):
    """divides score depending on true value"""
    if label==true_label:
        return score
    else:
        return -score
    
def add_predictions(data_text,data_label,model,tokenizer):
    encoded = tokenizer(data_text,padding=True,truncation=True, max_length=512,return_tensors="pt").to("cpu")
    predicted = model(**encoded).logits.reshape(-1).tolist()
    if len(predicted)==1:
        return {"regression":predicted[0]}
    else:
        pred_label = np.argmax(predicted)
        odd = np.exp(predicted[pred_label])
        prob = odd/(1+odd)
        output = {
            "predicted_label":pred_label,
            "prediction_score":prob,
            "separate_score":separate(pred_label,data_label,prob),
            "regression":translate(pred_label,data_label,prob)}
        return output
    
#models
def get_model(config,label_dicts=None):
    """returns model for bert of distilbert"""
    if config=="distilbert" or config=="distilbert-base-uncased":
        name="distilbert-base-uncased"
    elif config=="bert" or config=="bert-base-uncased":
        name="bert-base-uncased"
    else:
        name=config
    if label_dicts:
        id2label, label2id, num_labels = label_dicts[0], label_dicts[1], len(label_dicts[0].values())
        model = AutoModelForSequenceClassification.from_pretrained(name,id2label=id2label,label2id=label2id,num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(name)
    return model

def reload_model(model_dir,label_dicts=None):
    """reloads saved model at output_dir"""
    if label_dicts:
        id2label, label2id, num_labels = label_dicts[0], label_dicts[1], len(label_dicts[0].values())
        model = AutoModelForSequenceClassification.from_pretrained(model_dir,id2label=id2label,label2id=label2id,num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model

def get_generator(task,model,tokenizer):
    """returns generator for task"""
    return pipeline(task=task,model=model,tokenizer=tokenizer,truncation=True)


#training
def compute_metrics(eval_pred,is_hcl=False):
    """computes metrics for classification tasks"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    if is_hcl:
        acc = hcl.acc_fct(labels,predictions)
    else:
        acc = metric.compute(predictions=predictions, references=labels)["accuracy"]
    return {"acc":round(acc,4),"mse":round(mean_squared_error(labels, predictions),4)}
    
def train_model(model,output_dir,train_ds,test_ds,seed=42,num_train_epochs=3,is_hcl=False):
    """trains model"""
    training_args = TrainingArguments(output_dir=output_dir,seed=seed,num_train_epochs=num_train_epochs)
    trainer = Trainer(model=model,args=training_args,train_dataset=train_ds,eval_dataset=test_ds,compute_metrics = lambda x: compute_metrics(x,is_hcl=is_hcl))
    trainer.train()
    model.save_pretrained(output_dir)
    metrics = trainer.evaluate()
    return round(metrics["eval_acc"],4)


#analysis
def get_delay(T1):
    """returns deay after T1"""
    T2 = perf_counter()
    print("--------------------")
    print(f"Done in {(T2-T1)/60:.2f} min")
    print(" ")
    
def print_red(value_list,threshold):
    """return string values in red in higher than threshokd"""
    to_print =""
    for value in value_list:
        if value>=threshold:
            to_print += " "+f"\x1b[31m{value}\x1b[0m"
        else:
            to_print += " "+str(value)
    return to_print

def select_by_index(L,indexes):
    """returns L[indexes]"""
    Lbis = []
    for idx in indexes:
        Lbis.append(L[idx])
    return Lbis


    
#debug

def nested_shape(L):
    """prints shape of nested list"""
    if type(L[0])!=type(L):
        return [len(L)]
    else:
        shapes = []
        for sublist in L:
            shapes.append(len(sublist))
        return shapes