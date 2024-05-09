## mlm -- Gaspard Berthelier
# utilities for mlm models

#imports
import numpy as np
from time import perf_counter
import utils
from torch import Tensor
from transformers import DataCollatorForLanguageModeling, Trainer
import mia
from sklearn.metrics import classification_report
import data_hcl
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import ner


#mlm training
def compute_perplexity(model,tokenizer,train_ds,test_ds):
    """computes perplextiy for mlm model"""
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    trainer = Trainer(model=model,train_dataset=train_ds,eval_dataset=test_ds,data_collator=data_collator)
    eval_loss = trainer.evaluate()
    return round(math.exp(eval_loss['eval_loss']),3)

def train_mlm(model,tokenizer,train_ds,test_ds,output_dir,seed=42,num_train_epochs=3):
    """training function for mlm"""
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    training_args = TrainingArguments(output_dir=output_dir,seed=seed)
    trainer = Trainer(model=model,args=training_args,train_dataset=train_ds,eval_dataset=test_ds,data_collator=data_collator)
    trainer.train()
    model.save_pretrained(output_dir)
    eval_loss = trainer.evaluate()
    return round(math.exp(eval_loss['eval_loss']),4)


#likelihood
def add_mask(decomposed_text,mask_idx,mask="<mask>"):
    """adds a mask to extended text, returns extended masked text"""
    decomposed_masked_text = decomposed_text.copy()
    decomposed_masked_text[mask_idx]=mask
    return decomposed_masked_text

def extend_words(text,extended_words):
    """returns word idx of each tokens"""
    words = text.split(" ")
    words_idx = [None]
    j = 1
    skipped = False
    for i,word in enumerate(words):
        low_word = word.lower()
        step=True
        while step and j < len(extended_words)-1:
            temp_subword = extended_words[j]
            if "_" not in low_word:
                subword = ner.add_spaces(temp_subword).lower()
            else:
                subword = temp_subword.lower()
            if subword in low_word:
                words_idx.append(i)
                j+=1
            elif subword=="<unk>":
                words_idx.append(None)
                j+=1
            else:
                step=False
                
    words_idx.append(None)
    return words_idx

def recompose_mask(text,extended_text,decomposed_mask_text):
    """return masked text from decomposed masked text """
    words_idx = extend_words(text,extended_text)
    mask_words = []
    sentence = []
    for i,subword in enumerate(decomposed_mask_text):
        if i!=0 and i<len(words_idx)-1:
            if i!=1 and words_idx[i]==None:
                if subword=="<mask>":
                    mask_words.append(["<mask>"])
                else:
                    mask_words.append(["<unk>"])
            elif i!=1 and words_idx[i]==words_idx[i-1]:
                mask_words[-1].append(subword)
            else:
                mask_words.append([])
                mask_words[-1].append(subword)
    for word in mask_words:
        sentence.append("".join(word))
    return " ".join(sentence)

def get_mask_dataset(data,tokenizer,mask="<mask>"):
    seq = data["text"]
    dec = ner.get_words(tokenizer,seq) #extended words
    ds_dict = {"extended_text":dec}
    n = len(dec)
    n_masks = min(int(0.15*n),10)
    for k in range(n_masks):
        new_seq = add_mask(dec,np.random.randint(0,n),mask)
        ds_dict[f"extended_mask_{k}"] = new_seq
        ds_dict[f"masked_text_{k}"] = recompose_mask(seq,dec,new_seq)
    for k in range(n_masks,10):
        ds_dict[f"extended_mask_{k}"] = None
        ds_dict[f"masked_text_{k}"] = None
    #return {"extended_text":dec,"extended_masked_texts":extended_masks,"masked_texts":masks}
    return ds_dict
    
def add_dataset_masks(tokenizer,dataset):
    new_dataset = dataset.map(lambda data: get_mask_dataset(data,tokenizer))
    return new_dataset
                          
def compute_1_performance(model,masked_text,text,tokenizer,mask_token):
    """computes logit metric for masked_text compared to text """
    inputs = tokenizer(masked_text,padding="max_length",truncation=True,max_length=512,return_tensors="pt")
    labels = tokenizer(text,padding="max_length",truncation=True,max_length=512)["input_ids"]
    mask_idx = np.where(np.array(inputs["input_ids"][0].tolist())==mask_token)[0]
    logits = model(**inputs).logits[0].detach().numpy()
    
    psum = 0
    tot = 0
    for idx in mask_idx:
        true_word_id = labels[idx]
        true_word_logit = logits[idx][true_word_id]
        odd = np.exp(true_word_logit)
        psum += odd/(1+odd)
        tot += 1
    if tot != 0:
        return psum/tot
    else:
        None

def compute_performance(model,dataset,tokenizer,mask="<mask>"):
    """computes metric for each possible mask sequence and sums"""
    mask_token = tokenizer(mask)["input_ids"][1]
    N = dataset.num_rows
    perfs = []
    for k in range(N):
        esum = 0
        dec = dataset[f"extended_text"][k]
        n_new = min(int(0.15*len(dec)),10)
        n = n_new
        for j in range(n_new):
            perf = compute_1_performance(model,dataset[f"masked_text_{j}"][k],dataset["text"][k],tokenizer,mask_token)
            if perf:
                esum += perf
            else:
                n -= 1
        if n:
            perfs.append(round(esum/n,4))
        else:
            perfs.append(None)
    return perfs
                          
def remove_none(L):
    new_L = []
    for element in L:
        if element:
            new_L.append(element)
    return new_L

def compare_performance(model,model0,dataset,tokenizer,mask="<mask>"):
    """compares energy of model and base model for a given sequence"""
    e_target = compute_performance(model, dataset, tokenizer,mask=mask)
    e_base = compute_performance(model0, dataset, tokenizer,mask=mask)
    diff = [round(e_target[k]-e_base[k],4) if e_target[k] and e_base[k] else None for k in range(dataset.num_rows)]
    diff = np.array(remove_none(diff)).astype(float)
    return diff


def mia_pipeline(model,model0,dataset_train,dataset_test,tokenizer,mask,seed,save_dir,do_perfs=True,simple=True,pos_weight=1):
    t1 = perf_counter()
    if do_perfs:
        print("Train perfs")
        perfs_train = compare_performance(model,data_hcl.get_model("camembert-base"),dataset_train["text"],tokenizer,mask=mask)
        np.savetxt(f"{save_dir}/perfs_train.txt",perfs_train)
        print("Test perfs")
        perfs_test = compare_performance(model,data_hcl.get_model("camembert-base"),dataset_test["text"],tokenizer,mask=mask)
        np.savetxt(f"{save_dir}/perfs_test.txt",perfs_test)
        utils.get_delay(t1)
    else:
        perfs_train = np.loadtxt(f"{save_dir}/perfs_train.txt")
        perfs_test = np.loadtxt(f"{save_dir}/perfs_test.txt")

    np.random.seed(seed)
    np.random.shuffle(perfs_train)
    np.random.seed(seed)
    np.random.shuffle(perfs_test)
    
    n_train = len(perfs_train)
    n_test = len(perfs_test)
    M = min(n_train,n_test)-1
    m = int(0.9*M)
    print(f"Training MIA on {m} 'in's")
    print(f"Validation on {M-m} 'in's")
    
    plt.figure()
    sns.kdeplot([perfs_train,perfs_test])
    plt.xlabel("Performances compared to baseline")
    plt.legend(["Train","Test"])
    plt.xlim([-1.5,1.5])

    perfs_train = list(perfs_train[0:M])
    perfs_test = list(perfs_test[0:M])
            
    X_train = []
    y_train = []
    X_train = X_train + perfs_train[0:m]
    y_train = y_train + [1 for k in range(m)]
    X_train = X_train + perfs_test[0:m]
    y_train = y_train + [0 for k in range(m)]
    
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)
        
    print("Training MIA")
    if simple:
        xgb_model = xgb.XGBClassifier(max_depth=1,max_leaves=2,scale_pos_weight=pos_weight,n_jobs=-1).fit(np.array(X_train).reshape((-1,1)),np.array(y_train).reshape((-1,1)))
    else:
        xgb_model = xgb.XGBClassifier(scale_pos_weight=pos_weight,n_jobs=-1).fit(np.array(X_train).reshape((-1,1)),np.array(y_train).reshape((-1,1)))
    
    print("Predicting")
    print("Validation accuracy")
    true = [1 for k in range(m,M)]+[0 for k in range(m,M)]
    x = perfs_train[m:]+perfs_test[m:]
    pred = xgb_model.predict(np.array(x).reshape((-1,1)))
    utils.get_delay(t1)
    print(classification_report(true,pred))
    
    print("Training accuracy")
    Xpred = xgb_model.predict(np.array(X_train).reshape((-1,1)))
    utils.get_delay(t1)
    print(classification_report(y_train,Xpred))
    return true,pred,x
    

#extr
def compute_1_extraction(generator,decomposed_text,decomposed_masked_text,masked_text,mask="<mask>"):
    """checks if generator can predict masked token"""
    mask_idx = np.where(np.array(decomposed_masked_text)==mask)[0][0]
    try:
        output = generator(masked_text)
    except Exception as ex:
        print("pb")
        print(mask_idx)
        print(masked_text)
        if mask_idx:
            print(decomposed_masked_text[0:mask_idx+2])
            print(decomposed_text[0:mask_idx+2])
        print(ex)
    true_words = decomposed_text[mask_idx].split(" ")
    predicted_word = output[0]["token_str"]
    return len(predicted_word)>1 and predicted_word in true_words, true_words


                          
def get_private_extends(seq,ner_generator,ner_tokenizer,mask="<mask>"):
    out = ner_generator(seq) #pos labels
    dec = ner.get_words(ner_tokenizer,seq) #extendded words
    labels_idx = [entity["index"] for entity in out]
    labels = [1 if k in labels_idx else 0 for k in range(len(dec))]
    n = len(dec)
    masks = []
    extended_masks = []
    for k in range(n):
        if labels[k]==1:
            new_seq = add_mask(dec,k,mask=mask)
            extended_masks.append(new_seq)
            masks.append(recompose_mask(seq,dec,new_seq))
            
    ds_dict = {"extended_text":dec,"extended_labels":str(labels)}
    for k in range(10):
        if k<len(masks):
            ds_dict[f"extended_mask_{k}"] = extended_masks[k]
            ds_dict[f"masked_text_{k}"] = masks[k]
        else:
            ds_dict[f"extended_mask_{k}"] = None
            ds_dict[f"masked_text_{k}"] = None
    ds_dict[f"num_masks"]=min(10,len(masks))
    return ds_dict           
              
                          
def add_private_predictions(ner_generator,ner_tokenizer,dataset,mask="<mask>"):
    dataset = dataset.map(lambda data: get_private_extends(data["text"],ner_generator,ner_tokenizer,mask=mask))
    return dataset

def private_extraction(generator,dataset,mask="<mask>"):
    trues = 0
    total = 0
    extracted = []
    n = dataset.num_rows
    for k in range(n):
        num_masks = dataset["num_masks"][k]
        for j in range(num_masks):
            out = compute_1_extraction(generator,dataset["extended_text"][k],dataset[f"extended_mask_{j}"][k],dataset[f"masked_text_{j}"][k],mask=mask)
            if out[1] not in ["-","_"," ","",":",",",";",".","</s>"]:
                total+=1
                if out[0]:
                    trues+=1
                    extracted.append((dataset[f"extended_mask_{j}"][k],out[1]))
    return trues,total,extracted
                          
                          
#dump
                          
'''
def add_private_masks(decomposed_text,mask_idx,mask="<mask>"):
    """adds a mask to extended text"""
    decomposed_masked_text = decomposed_text.copy()
    decomposed_masked_text[mask_idx]=mask
    return decomposed_masked_text

def extend_words(text,extended_words):
    """returns word idx of each tokens"""
    words = text.split(" ")
    words_idx = [None]
    j = 1
    c = 0
    for i,word in enumerate(words):
        step=0
        while step==0 and j < len(extended_words)-1:
            temp_subword = extended_words[j]
            subword = add_spaces(temp_subword)
            if subword in word:
                words_idx.append(c)
                j+=1
            else:
                c+=1
                step=1
    words_idx.append(None)
    return words_idx

def recompose_mask(text,extended_text,decomposed_mask_text):
    """return masked text from decomposed masked text """
    words_idx = extend_words(text,extended_text)
    mask_words = []
    sentence = []
    for i,subword in enumerate(decomposed_mask_text):
        if i!=0 and i<len(words_idx)-1:
            if i!=1 and words_idx[i]==words_idx[i-1]:
                mask_words[-1].append(subword)
            else:
                mask_words.append([])
                mask_words[-1].append(subword)
    for word in mask_words:
        sentence.append("".join(word))
    return " ".join(sentence),sentence
'''                          
                          
''''                          
def add_mask(text,position,mask="<mask>",extended=False):
    """adds a mask on the text at position"""
    if extended:
        words = text.copy()
    else:
        words = text.split(" ")
    if position<len(words):
        new_words = words[0:position]+[mask]+words[position+1:]
    else:
        new_words = words[0:position]+[mask]
    return " ".join(new_words)

def create_masked_sequences(sequence,n_new,mask="<mask>",extended=False):
    """creates list a masked sequence with each word masked once"""
    masked = []
    for i in range(n_new):#number of masked sentences to create
        new_seq = add_mask(sequence,np.random.randint(0,num_words),mask,extended=extended)
        masked.append(new_seq)
    return masked

def get_mask_idx(sentence, mask = "<mask>", word_type="word"):
    """returns index of masked word"""
    idx = []
    if word_type=="word":
        for i,word in enumerate(sentence.split(" ")):
            if word == mask:
                idx.append(i)
    else:
        for i,token in enumerate(sentence):
            if token == mask:
                idx.append(i)
    return idx
'''

'''
def get_perfs(model,model0,sequences,tokenizer,mask):
    t1 = perf_counter()
    perfs = [compare_performance(model,model0,sequence,tokenizer,mask=mask) for sequence in sequences]
    perfs = [perf if perf else None for perf in perfs]
    utils.get_delay(t1)
    return perfs
'''