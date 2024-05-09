## fed -- Gaspard Berthelier
# federated learning applications

#imports

#technical
import torch
from torch.utils.data import DataLoader, SequentialSampler
from datasets import DatasetDict

#gaspard
import dp
import utils
import energy

#usuals
from tqdm.notebook import tqdm
import numpy as np
import copy as copy
from time import perf_counter

#datasets
def get_split_datasets(dataset,fractions,seed=42):
    """returns split dataset according to fractions"""
    dataset = dataset.shuffle(seed=seed)
    n = len(fractions)
    size = dataset.num_rows
    sizes = [int(fractions[i]*size) for i in range(n)]
    dataset_list = []
    rank = 0
    for i in range(n):
        if i<n-1:
            dataset_list.append(dataset.select(range(rank,rank+sizes[i])))
            rank = sizes[i]
        else:
            dataset_list.append(dataset.select(range(rank,size)))
    
    return [dataset_list[k] for k in range(n)]

def get_dataset_sizes(dataset_list):
    """prints split dataset sizes"""
    sizes = [dataset_list[k].num_rows for k in range(len(dataset_list))]
    return sizes, sum(sizes)


#training 
class LocalUpdate(object):
    """class for local model functions"""
    def __init__(self,train_ds,test_ds,bs,epoch=1,n_logs=4):
        self.ep = epoch #num local training epochs
        self.trainloader, self.testloader = DataLoader(train_ds,batch_size=bs), DataLoader(test_ds,batch_size=bs)
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bs = bs
        self.n_logs = n_logs
    
    def accuracy(preds, labels):
        return (preds == labels).mean()
    def inference(self, model):
        model.eval()
        loss_arr = []
        accuracy_arr = []
        for batch in self.testloader:
            batch = tuple(t.to(self.device) for t in batch)

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
                accuracy_arr.append(dp.accuracy(preds, labels))    
        model.train()
        return np.mean(accuracy_arr),np.mean(loss_arr)

    def update_weights(self, model):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4) #default lr=5e-4
        LOGGING_INTERVAL = int((len(self.trainloader.dataset)/self.bs)/self.n_logs) #number of logs per epoch is n_logs

        eval_accs = []
        for i in range(self.ep):
            losses = []
            for step, batch in enumerate(tqdm(self.trainloader)):

                optimizer.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                          'token_type_ids': batch[2],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                outputs = model(**inputs)
                loss = outputs[0]
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                
                if step > 0 and step % LOGGING_INTERVAL == 0:
                    train_loss = np.mean(losses)
                    eval_accuracy, eval_loss = self.inference(model)

                    print(
                      f"Epoch: {i} | "
                      f"Step: {step} | "
                      f"Train loss: {train_loss:.3f} | "
                      f"Eval loss: {eval_loss:.3f} | "
                      f"Eval accuracy: {eval_accuracy:.3f} | "
                    )
                    eval_accs.append(eval_accuracy)
                
        return model.state_dict(), eval_accs
    
def average_weights(w):
    """return average weights (no ponderations)"""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def train_fed(output_dir,dataset_list,global_model,batch_size=8,num_epochs=10,n_logs=4):
    """trains model with federated scheme"""
    trainable_layers = [global_model.bert.encoder.layer[-1], global_model.bert.pooler, global_model.classifier]
    total_params = 0
    trainable_params = 0
    n_part = len(dataset_list)
    for p in global_model.parameters():
            p.requires_grad = False
            total_params += p.numel()
    for layer in trainable_layers:
        for p in layer.parameters():
            p.requires_grad = True
            trainable_params += p.numel()
    
    for k in range(n_part):
        dataset_list[k]["train"]=dp.features_to_dataset(dataset_list[k]["train"])
        dataset_list[k]["test"]=dp.features_to_dataset(dataset_list[k]["test"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    global_model.train()
    global_weights = global_model.state_dict()

    epochs = num_epochs
    idxs_users = [k for k in range(n_part)]

    local_train_accs = []
    global_train_accs = []

    for epoch in range(epochs):
        print(f"Local epoch :  {epoch}")
        local_weights, local_accs = [], []
        global_model.train()

        for idx in idxs_users:
            print(f"Local user :  {idx}")
            local_model = LocalUpdate(dataset_list[idx]["train"],dataset_list[idx]["test"],batch_size,epoch=1,n_logs=n_logs) #1 local epoch
            w, accs = local_model.update_weights(model=copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w))
            local_accs.append(copy.deepcopy(accs)[-1]) #last local accuracy

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        print("Local accuracies for current epoch : ",local_accs)
        acc_avg = sum(local_accs) / len(local_accs) #average local (last) accuracies
        local_train_accs.append(local_accs) #average local accuracies for each epoch, before averaging

        # Calculate avg training accuracy over all users at every epoch
        list_acc = []
        global_model.eval()
        for idx in idxs_users:
            local_model = LocalUpdate(dataset_list[idx]["train"],dataset_list[idx]["test"],batch_size,epoch=epochs,n_logs=n_logs)
            acc,loss = local_model.inference(model=global_model) #accuracy after averaging
            list_acc.append(acc)
        print("Local accuracies for current epoch with FedAvg : ",list_acc)
        global_train_accs.append(list_acc) #average accuracy after averaging
    torch.save(global_model.state_dict(),f"{output_dir}/fed_model")
    return global_train_accs


def train_no_fed(output_dir,dataset,model,batch_size=8,num_epochs=10,n_logs=4):
    """trains model without federated learning"""
    
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
    train_dataloader = DataLoader(dataset["train"], batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(dataset["test"], sampler=SequentialSampler(dataset["test"]), batch_size=BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4) #default lr=5e-4

    EPOCHS = num_epochs
    LOGGING_INTERVAL = int((len(train_dataloader.dataset)/BATCH_SIZE)/n_logs) #number of logs per epoch is n_logs

    eval_accs = []
    for epoch in range(1, EPOCHS+1):
        print(f"Epoch : {epoch}")
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
                eval_loss, eval_accuracy = dp.evaluate(model,test_dataloader,device)

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



def fed_pipeline(dataset,fractions,label_dicts,build_data=True,train_vanilla=True,train_nodes_vanilla=True,do_train_fed=True,seed=42,save_dir="fed",model_name="bert",node=None,site=None):
    """pipeline to train federated classification """
    n_part = len(fractions)
    
    if build_data:
        print("Building dataset")
        t1 = perf_counter()
        dataset_list = get_split_datasets(dataset,fractions,seed=seed)
        sizes, total_size = get_dataset_sizes(dataset_list)
        print("Datasets sizes : ",sizes)
        print("Total size : ",total_size)
        tokenizer_dp = dp.get_dp_tokenizer(model_name)
        for k in range(len(dataset_list)):
            dataset_list[k]=dataset_list[k].train_test_split(test_size=0.1,seed=seed)
        utils.get_delay(t1)
    else:
        dataset_list = dataset
        
    if train_vanilla:
        print("Training vanilla model")
        normal_model = dp.get_dp_model(model_name,label_dicts)
        train_test = dataset.train_test_split(test_size=0.1,seed=seed)
        train = dp.features_to_dataset(train_test["train"])
        test = dp.features_to_dataset(train_test["test"])
        train_test = DatasetDict({"train":train,"test":test})
        
        t1 = perf_counter()
        dt1 = energy.get_time()
        accs_vanilla = train_no_fed(f"{save_dir}",train_test,normal_model,batch_size=8,num_epochs=10)
        t2 = perf_counter()
        delay = (t2-t1)/60
        dt2 = energy.get_time()
        mean_energy, total_energy = energy.get_power_info(node, site, dt1, dt2, metric="bmc_node_power_watt")
        
        np.savetxt(f"{save_dir}/vanilla_accs.txt",accs_vanilla)
        np.savetxt(f"{save_dir}/vanilla_time.txt",[delay])
        np.savetxt(f"{save_dir}/vanilla_mean_energies.txt",[mean_energy])
        np.savetxt(f"{save_dir}/vanilla_total_energies.txt",[total_energy])
    else:
        accs_vanilla = np.loadtxt(f"{save_dir}/vanilla_accs.txt")
        mean_energy = np.loadtxt(f"{save_dir}/vanilla_mean_energies.txt")
        total_energy = np.loadtxt(f"{save_dir}/vanilla_total_energies.txt")
        delay = np.loadtxt(f"{save_dir}/vanilla_time.txt")
    print("")
    print("Global vanilla")
    print("Final vanilla accuracy : ",round(accs_vanilla[-1],4))
    print(f"Training time : {delay:.2f} min")
    print("Mean energy (W) : ",mean_energy)
    print("Total energy (kW) : ",total_energy)
    print("")
    
    if train_nodes_vanilla:
        print("Training vanilla nodes")
        for i in range(n_part):
            normal_model = dp.get_dp_model(model_name,label_dicts)
            train = dp.features_to_dataset(dataset_list[k]["train"])
            test = dp.features_to_dataset(dataset_list[k]["test"])
            train_test = DatasetDict({"train":train,"test":test})
    
            print(f"Training vanilla model {i}")
            t1 = perf_counter()
            dt1 = energy.get_time()
            local_acc = train_no_fed(f"{save_dir}/model_{i}",train_test,normal_model,batch_size=8,num_epochs=10)
            t2 = perf_counter()
            delay = (t2-t1)/60
            dt2 = energy.get_time()
            mean_energy, total_energy = energy.get_power_info(node, site, dt1, dt2, metric="bmc_node_power_watt")
    
            np.savetxt(f"{save_dir}/model_{i}/vanilla_accs_node_{i}.txt",[round(local_acc[-1],4)])
            np.savetxt(f"{save_dir}/model_{i}/vanilla_time_node_{i}.txt",[round(delay,2)])
            np.savetxt(f"{save_dir}/model_{i}/vanilla_mean_energies_node_{i}.txt",[mean_energy])
            np.savetxt(f"{save_dir}/model_{i}/vanilla_total_energies_node_{i}.txt",[total_energy])
    local_accs = []
    mean_energies = []
    total_energies = []
    delays = []
    for i in range(n_part):
        local_accs.append(float(np.loadtxt(f"{save_dir}/model_{i}/vanilla_accs_node_{i}.txt")))
        mean_energies.append(float(np.loadtxt(f"{save_dir}/model_{i}/vanilla_mean_energies_node_{i}.txt")))
        total_energies.append(float(np.loadtxt(f"{save_dir}/model_{i}/vanilla_total_energies_node_{i}.txt")))
        delays.append(float(np.loadtxt(f"{save_dir}/model_{i}/vanilla_time_node_{i}.txt")))
    print("")
    print("Local vanilla")
    print("Final vanilla accuracies : ",local_accs)
    print(f"Training times (min) : {delays}")
    print("Mean energies (W) : ",mean_energies)
    print("Total energies (kW) : ",total_energies) 
    
    if do_train_fed:
        print("Training fed model")
        t1 = perf_counter()
        dt1 = energy.get_time()
        fed_model = dp.get_dp_model(model_name,label_dicts)
        fed_accs = train_fed(f"{save_dir}",dataset_list,fed_model,batch_size=8,num_epochs=10,n_logs=4)
        t2 = perf_counter()
        delay = (t2-t1)/60
        dt2 = energy.get_time()
        mean_energy, total_energy = energy.get_power_info(node, site, dt1, dt2, metric="bmc_node_power_watt")
    
        np.savetxt(f"{save_dir}/fed_global_train_accs.txt",fed_accs)
        np.savetxt(f"{save_dir}/fed_time.txt",[delay])
        np.savetxt(f"{save_dir}/fed_mean_energies.txt",[mean_energy])
        np.savetxt(f"{save_dir}/fed_total_energies.txt",[total_energy])
    else:
        fed_accs = np.loadtxt(f"{save_dir}/fed_global_train_accs.txt")
        delay = np.loadtxt(f"{save_dir}/fed_time.txt")
        mean_energy = np.loadtxt(f"{save_dir}/fed_mean_energies.txt")
        total_energy = np.loadtxt(f"{save_dir}/fed_total_energies.txt")
    print("")
    print("Fed")
    print("Local accuracies for average model : ",[round(fed_accs[-1][k],4) for k in range(n_part)])
    print(f"Training time :  {delay:.2f} min")
    print("Mean energy : ",mean_energy)
    print("Total energy : ",total_energy)
    
    return dataset_list,accs_vanilla,local_accs,fed_accs