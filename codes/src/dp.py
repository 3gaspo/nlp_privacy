## dp - Gaspard Berthelier
# differential privacy applications

##imports

#nlp
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from datasets import concatenate_datasets, DatasetDict

#gaspard
import utils
import energy
import mia

#usual
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from time import perf_counter

#dataset
def get_dp_tokenizer(text):
    """returns tokenizer for dp training"""
    if text=="distilbert":
        name = "distilbert-base-uncased"
    elif text=="bert":
        name = "bert-base-uncased"
    else:
        name=text
        
    if name in ["bert-base-uncased","bert-base-cased"]:
        return BertTokenizer.from_pretrained(name,do_lower_case=False)
    elif name in ["distilbert-base-uncased","distilbert-base-cased","distilbert"]:
        return DistilBertTokenizer.from_pretrained(name,do_lower_case=False)
    else:
        print("Model not supported")
        
def features_to_dataset(features):
    """adapts tokenized dataset to pytorch format"""
    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
    all_labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def get_dp_datasets(dataset,tokenizer,test_size=0.1,seed=42,split=None):
    """returns train and test datasets (if do_split, else returns tokenized dataset) for DP training"""
    tokenized_dataset = utils.tokenize_dataset(dataset,tokenizer)
    if "label" not in dataset.column_names:
        if "true_label" in dataset.column_names:
            tokenized_dataset = tokenized_dataset.rename_column("true_label","label")
        else:
            print("No labels found")
    if split:
        if split=="split":
            main_ds = tokenized_dataset.train_test_split(test_size=0.1,seed=seed)
            train_ds = features_to_dataset(main_ds["train"])
            test_ds = features_to_dataset(main_ds["test"])
        else:
            if "status" in dataset.column_names:
                train_ds = features_to_dataset(tokenized_dataset.filter(lambda data: data["status"]==1))
                test_ds = features_to_dataset(tokenized_dataset.filter(lambda data: data["status"]==0))
            else:
                print("No train/test separation found")
        return train_ds, test_ds
    else:
        return features_to_dataset(tokenized_dataset)


##model
def get_dp_model(config_name,label_dicts=None):
    """returns model adapted for dp training (for classification task)"""
    if config_name=="distilbert":
        name = "distilbert-base-uncased"
    elif config_name=="bert":
        name = "bert-base-uncased"
    else:
        name=config_name
    if name in ["bert-base-uncased","bert-base-cased"]:
        if label_dicts:
            id2label, label2id, num_labels = label_dicts[0], label_dicts[1], len(label_dicts[0].values())
            config = BertConfig.from_pretrained(name,id2label=id2label,label2id=label2id,num_labels=num_labels)
        else:
            config = BertConfig.from_pretrained(name)
        return BertForSequenceClassification.from_pretrained(name,config=config)
    
    elif name in ["distilbert-base-uncased","distilbert-base-cased","distilbert"]:
        if label_dicts:
            id2label, label2id, num_labels = label_dicts[0], label_dicts[1], len(label_dicts[0].values())
            config = DistilBertConfig.from_pretrained(name,id2label=id2label,label2id=label2id,num_labels=num_labels)
        else:
            config = DistilBertConfig.from_pretrained(name)
        return DistilBertForSequenceClassification.from_pretrained(name,config=config)
    else:
        print("Model not supported")
    
def reload_dp_model(config_name,model_dir,label_dicts=None,model_type="dp"):
    """reloads dp model, model_type either dp or vanilla"""
    id2label, label2id, num_labels = label_dicts[0], label_dicts[1], len(label_dicts[0].values())
    
    if config_name=="distilbert":
        name = "distilbert-base-uncased"
    elif config_name=="bert":
        name = "bert-base-uncased"
    else:
        name=config_name
        
    if name in ["bert-base-uncased","bert-base-cased"]:
        if label_dicts:
            id2label, label2id, num_labels = label_dicts[0], label_dicts[1], len(label_dicts[0].values())
            config = BertConfig.from_pretrained(name,id2label=id2label,label2id=label2id,num_labels=num_labels)
        else:
            config = BertConfig.from_pretrained(name)
        return BertForSequenceClassification.from_pretrained(name,config=config,state_dict = torch.load(f"{model_dir}/{model_type}_model"))
    
    elif name in ["distilbert-base-uncased","distilbert-base-cased","distilbert"]:
        if label_dicts:
            id2label, label2id, num_labels = label_dicts[0], label_dicts[1], len(label_dicts[0].values())
            config = DistilBertConfig.from_pretrained(name,id2label=id2label,label2id=label2id,num_labels=num_labels)
        else:
            config = DistilBertConfig.from_pretrained(name)
        return DistilBertForSequenceClassification.from_pretrained(name,config=config,state_dict = torch.load(f"{model_dir}/{model_type}_model"))
    else:
        print("Model not supported")


## training
def accuracy(preds, labels):
    """returns accuracy (for DP models)"""
    return (preds == labels).mean()

def evaluate(model, test_dataloader, device):
    """returns evaluation of current model"""
    model.eval()
    loss_arr = []
    accuracy_arr = []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            labels = inputs['labels'].detach().cpu().numpy()
            loss_arr.append(loss.item())
            accuracy_arr.append(accuracy(preds, labels))
    model.train()
    return np.mean(loss_arr), np.mean(accuracy_arr) #averages metrics of batches

def dp_train(train_dataset,test_dataset,model,target_eps,num_epochs,output_dir,max_grad_norm=1,noise=None,batch_size=8,learning_rate=5e-4,delta_multiplyer=1,batch_divide=1,n_logs=4):
    """trains model with DP, returns accuracies and epsilons"""
    trainable_layers = [model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
            p.requires_grad = False
            total_params += p.numel()
    for layer in trainable_layers:
        for p in layer.parameters():
            p.requires_grad = True
            trainable_params += p.numel()
    
    BATCH_SIZE = batch_size #plus grand => moins de bruit à chaque data => epsilon croit plus vite
    MAX_PHYSICAL_BATCH_SIZE = BATCH_SIZE/batch_divide #for computation optimization < batch size
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #default lr=5e-4
        
    EPOCHS = num_epochs #number of loops
    EPSILON = target_eps
    DELTA = (1 / len(train_dataloader))*delta_multiplyer
    NOISE = noise
    MAX_GRAD_NORM = max_grad_norm #trop bas => risque de perte de convergence #askip garder à 1 est bien
    #bruit ajouté : std=noise_multiplier * max_grad_norm    pour avoir le même target epsilon 
    privacy_engine = PrivacyEngine()
    if target_eps:
        model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            target_delta=DELTA,
            target_epsilon=EPSILON, 
            epochs=EPOCHS,
            max_grad_norm=MAX_GRAD_NORM,
        )
    if noise and not target_eps:
        model, optimizer, train_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            noise_multiplier=NOISE,
            max_grad_norm=MAX_GRAD_NORM,
        )
    eval_accs = []
    epsilons = []
    
    LOGGING_INTERVAL = int((len(train_dataloader.dataset)/MAX_PHYSICAL_BATCH_SIZE)/n_logs)
    print("starting DP training")
                                                                                 
    for epoch in range(1, EPOCHS+1):
        print(f"epoch {epoch}")
        losses = []
        with BatchMemoryManager(
            data_loader=train_dataloader, 
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for step, batch in enumerate(tqdm(memory_safe_data_loader)):
                optimizer.zero_grad()
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels':         batch[3]}
                outputs = model(**inputs) # output = loss, logits, hidden_states, attentions
                loss = outputs[0]
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

                if step > 0 and step % LOGGING_INTERVAL == 0:
                    train_loss = np.mean(losses)
                    eps = privacy_engine.get_epsilon(DELTA)
                    eval_loss, eval_accuracy = evaluate(model,test_dataloader,device)
                    print(
                      f"Epoch: {epoch} | "
                      f"Step: {step} | "
                      f"Train loss: {train_loss:.3f} | "
                      f"Eval loss: {eval_loss:.3f} | "
                      f"Eval accuracy: {eval_accuracy:.3f} | "
                      f"ɛ: {eps:.2f}"
                    )
                    eval_accs.append(eval_accuracy)
                    epsilons.append(eps)

    torch.save(model.state_dict(),f"{output_dir}/dp_model")
    return eval_accs,epsilons

def no_dp_train(train_dataset,test_dataset,model,num_epochs,output_dir,batch_size=8,n_logs=4):
    """train model without DP, returns accuracies"""
    trainable_layers = [model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
            p.requires_grad = False
            total_params += p.numel()
    for layer in trainable_layers:
        for p in layer.parameters():
            p.requires_grad = True
            trainable_params += p.numel()

    BATCH_SIZE = batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4) #default lr=5e-4
        
    EPOCHS = num_epochs
    LOGGING_INTERVAL = int((len(train_dataloader.dataset)/BATCH_SIZE)/n_logs) #number of logs per epoch is n_logs
    print("starting normal training")
    
    eval_accs = []
    for epoch in range(1, EPOCHS+1):
        losses = []
        for step, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs) # output = loss, logits, hidden_states, attentions
            loss = outputs[0]
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            if step > 0 and step % LOGGING_INTERVAL == 0:
                train_loss = np.mean(losses)
                #eps = privacy_engine.get_epsilon(DELTA)
                eval_loss, eval_accuracy = evaluate(model,test_dataloader,device)

                print(
                  f"Epoch: {epoch} | "
                  f"Step: {step} | "
                  f"Train loss: {train_loss:.3f} | "
                  f"Eval loss: {eval_loss:.3f} | "
                  f"Eval accuracy: {eval_accuracy:.3f} | "
                )
                eval_accs.append(eval_accuracy)
    torch.save(model.state_dict(),f"{output_dir}/vanilla_model")
             
    return eval_accs


#pipeline
def dp_pipeline(dataset,params,label_dicts,seed=42,build_data=True,train_vanilla=True,do_train_dp=True,do_plot=True,save_dir="dp",n_logs=4,trial=1,model_name="bert",test_size=0.1,node=None,site=None):
    """trains vanialla and dp models returns plot of accuracy"""
    
    if build_data:
        print("Building dataset")
        t1 = perf_counter()
        tokenizer_dp = get_dp_tokenizer(model_name)
        train_dp, test_dp = get_dp_datasets(dataset,tokenizer_dp,test_size=0.1,seed=seed,split="split")
        utils.get_delay(t1)
    else:
        train_dp, test_dp = dataset["train"], dataset["test"]
        
    target_eps,max_grad_norm,num_epochs,batch_size,learning_rate,noise,delta_multiplier,batch_divide = params.values()
    print(f"Current params : {params}")
    
    if train_vanilla:
        print("Training vanilla")
        t1 = perf_counter()
        dt1 = energy.get_time()
        model = get_dp_model(model_name,label_dicts=label_dicts)
        accs_vanilla = no_dp_train(train_dp,test_dp,model,num_epochs,save_dir,batch_size=batch_size,n_logs=n_logs)
        t2 = perf_counter()
        delay = (t2-t1)/60
        dt2 = energy.get_time()
        mean_energy, total_energy = energy.get_power_info(node, site, dt1, dt2, metric="bmc_node_power_watt")
        np.savetxt(f"{save_dir}/accs_vanilla.txt",accs_vanilla)
        np.savetxt(f"{save_dir}/vanilla_mean_energies.txt",[mean_energy])
        np.savetxt(f"{save_dir}/vanilla_total_energies.txt",[total_energy])
        np.savetxt(f"{save_dir}/vanilla_time.txt",[delay])
 
    else:
        accs_vanilla = np.loadtxt(f"{save_dir}/accs_vanilla.txt")
        if node and site:
            mean_energy = np.loadtxt(f"{save_dir}/vanilla_mean_energies.txt")
            total_energy = np.loadtxt(f"{save_dir}/vanilla_total_energies.txt")
            delay = np.loadtxt(f"{save_dir}/vanilla_time.txt")
        else:
            mean_energy = 0
            total_energy = 0
            delay = 0
    if do_plot:
        print("")
        print("Vanilla")
        print("Final accuracy : ",round(accs_vanilla[-1],4))
        print(f"Training time : {delay:.2f} min")
        print("Mean energy : ",mean_energy)
        print("Total energy : ",total_energy)
        
    if do_train_dp:
        print(f"Training DP")
        t1 = perf_counter()
        dt1 = energy.get_time()
        model = get_dp_model(model_name,label_dicts=label_dicts)
        accs_dp,eps_dp = dp_train(train_dp,test_dp,model,target_eps,num_epochs,save_dir,max_grad_norm=max_grad_norm,noise=noise,batch_size=batch_size,learning_rate=learning_rate,delta_multiplyer=delta_multiplier,batch_divide=batch_divide,n_logs=n_logs)
        t2 = perf_counter()
        delay = (t2-t1)/60
        dt2 = energy.get_time()
        mean_energy, total_energy = energy.get_power_info(node, site, dt1, dt2, metric="bmc_node_power_watt")
        np.savetxt(f"{save_dir}/accs_dp.txt",accs_dp)
        np.savetxt(f"{save_dir}/eps_dp.txt",eps_dp)
        np.savetxt(f"{save_dir}/dp_mean_energies.txt",[mean_energy])
        np.savetxt(f"{save_dir}/dp_total_energies.txt",[total_energy])
        np.savetxt(f"{save_dir}/dp_time.txt",[delay])

    else:
        accs_dp = np.loadtxt(f"{save_dir}/accs_dp.txt")
        eps_dp = np.loadtxt(f"{save_dir}/eps_dp.txt")
        if node and site:
            mean_energy = np.loadtxt(f"{save_dir}/dp_mean_energies.txt")
            total_energy = np.loadtxt(f"{save_dir}/dp_total_energies.txt")
            delay = np.loadtxt(f"{save_dir}/dp_time.txt")
        else:
            mean_energy = 0
            total_energy = 0
            delay = 0

    if do_plot:
        print("")
        print("DP")
        fig = plt.figure(figsize = (5,3))
        plt.plot(range(len(accs_vanilla)),accs_vanilla)
        plt.plot(range(len(accs_dp)),accs_dp)
        plt.title(f"Accuracies trial {trial}")
        plt.legend(["no DP","DP"])
        plt.savefig(f"{save_dir}/trial_{trial}_{params}.png")
        plt.show() 
        print("Acc : ",round(accs_dp[-1],4))
        print("Eps : ",round(eps_dp[-1],4))
        print(f"Training time : {delay:.2f} min")
        print("Mean energy : ",mean_energy)
        print("Total energy : ",total_energy)
    
    return accs_vanilla, accs_dp, eps_dp, train_dp, test_dp



def get_simple_dp(dataset_raw,seed=42,do_vanilla_training=True,do_dp_training=True):

    tokenizer_dp = get_dp_tokenizer("bert")
    dataset = dataset_raw.filter(lambda data: data["label"] in [0,6])
    dataset = dataset.map(lambda data: {"new_label": int(data["label"]==6)},remove_columns=["label"])
    dataset = dataset.rename_column("new_label","label")
    new_label_dicts = [{0: 'LABEL_0', 1: 'LABEL_1'}, {'LABEL_0': 0, 'LABEL_1': 1}]
    train_dp = dataset.filter(lambda data: data["status"]==1)
    test_dp = dataset.filter(lambda data: data["status"]==0)
    n_train, n_test = len(train_dp), len(test_dp)
    n_dp = min(n_train,n_test)
    train_dp = train_dp.select(range(n_dp))
    test_dp = test_dp.select(range(n_dp))
    print("Training size: ",len(train_dp))
   
    dataset = concatenate_datasets([train_dp,test_dp])
    train_dp = get_dp_datasets(train_dp,tokenizer_dp,test_size=0.1,seed=seed,split=None)
    test_dp = get_dp_datasets(test_dp,tokenizer_dp,test_size=0.1,seed=seed,split=None)
    params = {"target_eps":None,"max_grad_norm":1,"num_epochs":10,"batch_size":8,"learning_rate":5e-4,"noise":0.15,"delta_multiplier":1,"batch_divide":1}
   
    if do_vanilla_training:
        print("Training vanilla")
        model = get_dp_model("bert",new_label_dicts)
        t1 = perf_counter()
        accs_vanilla = no_dp_train(train_dp,test_dp,model,params["num_epochs"],"dp_mia",batch_size=params["batch_size"],n_logs=4)
        np.savetxt("dp_mia/accs_vanilla.txt",accs_vanilla)
        utils.get_delay(t1)
    else:
        accs_vanilla = np.loadtxt("dp_mia/accs_vanilla.txt")
    print("Vanilla acc : ",round(accs_vanilla[-1],4))
    
    
    if do_dp_training:
        t1 = perf_counter()
        model = get_dp_model("bert",new_label_dicts)
        accs_vanilla, accs_dp, eps_dp, train_dp, test_dp = dp_pipeline(DatasetDict({"train":train_dp,"test":test_dp}),params,new_label_dicts,seed=seed,build_data=False,train_vanilla=False,do_train_dp=do_dp_training,do_plot=True,save_dir="dp_mia",n_logs=4,trial=1,model_name="bert",test_size=0.1,node=None,site=None)
        np.savetxt("dp_mia/accs_dp.txt",accs_dp)
        np.savetxt("dp_mia/eps_dp.txt",eps_dp)
        utils.get_delay(t1)
    else:
        accs_dp = np.loadtxt("dp_mia/accs_dp.txt")
        eps_dp = np.loadtxt("dp_mia/eps_dp.txt")
    print("DP acc : ",round(accs_dp[-1],4))
    print("eps : ",round(eps_dp[-1],4))
    
    return dataset, tokenizer_dp, new_label_dicts
    