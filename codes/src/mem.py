## mem -- Gaspard Berthelier
#counterfactual memorization applications

#imports

#technical
from transformers import TrainingArguments,Trainer, pipeline
from sklearn.model_selection import ShuffleSplit

#gaspard
import utils
import patho

#usuals
import numpy as np
from tqdm.notebook import tqdm
from time import perf_counter
import seaborn as sns
import matplotlib.pyplot as plt


#dataset
def get_mem_splits(dataset,n_splits=20,random_state=42,test_size=0.25):
    """returns index splits for dataset"""
    rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    train_tests = []
    for train_index, test_index in rs.split(dataset):
        train_tests.append({"in":train_index,"out":test_index})
    return train_tests


#training
def train_for_memorization(n_models,dataset,config,label_dicts,output_dir="mem",seed=42,num_train_epochs=3,start_at=0,is_hcl=False):
    """trains n_models with different dataset splits, saves, model and returns split indexes"""
    train_tests = get_mem_splits(dataset,n_splits=n_models,random_state=seed,test_size=0.5)
    accs = [None for k in range(n_models)]
    for i in range(n_models):
        if i >= start_at:
            if is_hcl:
                new_model = patho.get_model(config,label_dicts=label_dicts)
            else:
                new_model = utils.get_model(config,label_dicts=label_dicts)
            print(f"Training model {i}")
            acc = utils.train_model(new_model,f"{output_dir}/model_{i}",dataset.select(train_tests[i]["in"]),dataset.select(train_tests[i]["out"]),seed=seed,num_train_epochs=num_train_epochs,is_hcl=is_hcl)
            acc=round(acc,4)
            accs[i]=acc
            print(acc)
        else:
            print(f"Skipping {i}")
    return train_tests, accs

#memorization
def get_performance(dataset,train_tests,tokenizer,models_dir="mem"):
    """returns performance info for each data"""
    n_models = len(train_tests)
    size = dataset.num_rows
    mem_info = [[[0,0],[0,0]] for k in range(size)] #[(perf_in,perf_out),(n_in,n_out)]

    for i in range(n_models):
        print(f"Computing memorization for model {i}")
        model = utils.reload_model(f"{models_dir}/model_{i}")
        for j,data in enumerate(tqdm(dataset["text"])):
            #compute performance
            encoded = tokenizer(data,padding=True,truncation=True,max_length=512,return_tensors="pt").to("cpu")
            predictions = model(**encoded).logits.reshape(-1).tolist()
            true_label = dataset["label"][j]
            odd = np.exp(predictions[true_label])
            perf = odd/(1+odd)
            #check in/out
            if j in train_tests[i]["out"]:
                mem_info[j][0][1] += perf
                mem_info[j][1][1] += 1
            else:
                mem_info[j][0][0] +=perf
                mem_info[j][1][0] += 1

    return mem_info

def get_memorization(mem_info):
    """returns computed memorization"""
    mems = [None for k in range(len(mem_info))]
    for j in range(len(mem_info)):
        if mem_info[j][1][0]!=0 and mem_info[j][1][1] !=0:
            mems[j] = mem_info[j][0][0]/mem_info[j][1][0]-mem_info[j][0][1]/mem_info[j][1][1] #perf_in/n_in - perf_out/n_out
    return mems

def get_mem_threshold(mems,threshold):
    """returns indexes of mems higher than threshold"""
    real_mems = []
    for j,mem in enumerate(mems):
        if mem and mem>threshold:
            real_mems.append(j)
    return real_mems


def mem_results(mems,dataset,hallmarks,means,real_t=0.3,print_results=True):
    """returns results of mem pipeline"""
    real_mems = get_mem_threshold(mems,real_t)
    
    if print_results:
        #memorization distribution
        fig = plt.figure()
        sns.displot(mems).set(title="Counterfactual memorization distribution")
        plt.xlim([-0.5,0.7])
        plt.show()
        
        #counterfactuals
        print(f"Number of counterfactuals for threshold : {real_t} : ",len(real_mems))
        mean_words, mean_uniques, mean_sizes = means
        real_mems_text = dataset.select(real_mems)["text"]
        real_mems_label = dataset.select(real_mems)["label"]
        mem_sizes = utils.get_size_count(real_mems_text)
        mem_uniques = utils.get_unique_count(real_mems_text)
        mem_words = utils.get_word_count(real_mems_text)
        mem_labels = [hallmarks[real_mems_label[k]] for k in range(len(real_mems))]
        print(f"Words with mem > {real_t}")
        print("-----------------")
        print(f"Sizes : {utils.print_red(mem_sizes,mean_sizes)}")
        print(f"Words : {utils.print_red(mem_words,mean_words)}")
        print(f"Unique words : {utils.print_red(mem_uniques,mean_uniques)}")
        print("Labels : ",mem_labels)
        
    return real_mems
    
def mem_pipeline(dataset,tokenizer,label_dicts,means,real_t=0.3,n_counters=10,do_train=True,do_perf=True,model_dir="mem",seed=42,num_train_epochs=3,print_results=True,model_name="bert",is_hcl=False):
    """pipeline to test memorization onf model type on dataset"""
    #training
    t1 = perf_counter()
    label_names = list(label_dicts[0].values())
    
    if do_train:
        train_tests,accs = train_for_memorization(n_counters,dataset,model_name,label_dicts,output_dir=model_dir,seed=seed,num_train_epochs=num_train_epochs,start_at=0,is_hcl=is_hcl) #300 min
        np.savetxt(f"{model_dir}/accs.txt",accs)
    if do_perf:
        if not do_train:
            train_tests = get_mem_splits(dataset,n_splits=n_counters,random_state=seed,test_size=0.5)
            accs = []
        mem_info = get_performance(dataset,train_tests,tokenizer,models_dir=model_dir)
        mems = get_memorization(mem_info)
        mems = [mem if mem else -100 for mem in mems]
        np.savetxt(f"{model_dir}/mems.txt",mems)
        mems = [mem if mem != -100 else None for mem in mems]
        utils.get_delay(t1)
    else:
        mems = np.loadtxt(f"{model_dir}/mems.txt")
        mems = [mem if mem != -100 else None for mem in mems ]
        accs = np.loadtxt(f"{model_dir}/accs.txt")
        if print_results:
            print("Accs : ",np.mean(accs)) 
    
    #distribution
    real_mems = mem_results(mems,dataset,label_names,means,real_t=real_t,print_results=print_results)
    return real_mems
    
    