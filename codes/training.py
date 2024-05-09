## training -- Gaspard Berthelier
# to train models

#imports

import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter
import energy
import patho
#overfit

def get_perf(true,pred,score):
    """returns a classifier's performance"""
    perfs = []
    for i,label in enumerate(true):
        if label==pred[i]:
            perfs.append(score[i])
        else:
            perfs.append(-score[i])
    return perfs

def compute_overfit(model,model0,train_ds,test_ds,tokenizer):
    """returns a classifier's performance compared to baseline"""
    generator0 = utils.get_generator("text-classification",model0,tokenizer)
    generator = utils.get_generator("text-classification",model,tokenizer)
    #model0
    train_perf_ds0 = train_ds.map(lambda data: utils.add_predictions(data["text"],data["label"],model0,tokenizer))
    test_perf_ds0 = test_ds.map(lambda data: utils.add_predictions(data["text"],data["label"],model0,tokenizer))
    train_perfs0 = get_perf(train_perf_ds0["label"],train_perf_ds0["predicted_label"],train_perf_ds0["prediction_score"])
    test_perfs0 = get_perf(test_perf_ds0["label"],test_perf_ds0["predicted_label"],test_perf_ds0["prediction_score"])
    #model
    train_perf_ds = train_ds.map(lambda data: utils.add_predictions(data["text"],data["label"],model,tokenizer))
    test_perf_ds = test_ds.map(lambda data: utils.add_predictions(data["text"],data["label"],model,tokenizer))
    train_perfs = get_perf(train_perf_ds["label"],train_perf_ds["predicted_label"],train_perf_ds["prediction_score"])
    test_perfs = get_perf(test_perf_ds["label"],test_perf_ds["predicted_label"],test_perf_ds["prediction_score"])
    return train_perfs, test_perfs, train_perfs0, test_perfs0
        
def plot_overfit(train_perfs, test_perfs, train_perfs0, test_perfs0):
    idx_true = np.where(np.array(train_perfs)>0)[0]
    idx_false = np.where(np.array(train_perfs)<=0)[0]
    idx_true_test = np.where(np.array(test_perfs)>0)[0]
    idx_false_test = np.where(np.array(test_perfs)<=0)[0]
    train_perf_pos = utils.select_by_index(train_perfs,idx_true)
    train_perf_neg = utils.select_by_index(train_perfs,idx_false)
    test_perf_pos = utils.select_by_index(test_perfs,idx_true_test)
    test_perf_neg = utils.select_by_index(test_perfs,idx_false_test)
    to_plot = {
        "Training performance":train_perf_pos,#negatives only has 1 won't work
        "Test performance":test_perfs,
        "Baseline train":train_perfs0,
        "Baseline test":test_perfs0}
    
    n = len(to_plot.values())
    fig, axes = plt.subplots(1,4,figsize=(20,n))
    binwidths = [0.0001,0.05,0.01,0.01]
    for k in range(n):
        sns.histplot(list(to_plot.values())[k],ax=axes[k],color="red",binwidth=binwidths[k]).set(title=list(to_plot.keys())[k])
    axes[0].set_xlim([0.99,1.001])
    #axes[2].set_xlim([-0.5,0.5])
    #axes[3].set_xlim([-0.5,0.5])
    fig.tight_layout()
    
def get_learning(model,model0,train_ds,test_ds,tokenizer,save_dir="target_model",save=True,redo=True,plot_distrib=True,means=True):
    """returns performance value"""
    if redo:
        train_perfs, test_perfs, train_perfs0, test_perfs0 = compute_overfit(model,model0,train_ds,test_ds,tokenizer)
        if save:
            np.savetxt(f"{save_dir}/train_perfs.txt",train_perfs)
            np.savetxt(f"{save_dir}/test_perfs.txt",test_perfs)
            np.savetxt(f"{save_dir}/train_perfs0.txt",train_perfs0)
            np.savetxt(f"{save_dir}/test_perfs0.txt",test_perfs0)
    else:
        train_perfs = np.loadtxt(f"{save_dir}/train_perfs.txt")
        test_perfs = np.loadtxt(f"{save_dir}/test_perfs.txt")
        train_perfs0 = np.loadtxt(f"{save_dir}/train_perfs0.txt")
        test_perfs0 = np.loadtxt(f"{save_dir}/test_perfs0.txt")
    
    if plot_distrib:
        plot_overfit(train_perfs, test_perfs, train_perfs0, test_perfs0)
    
    learnings_train = np.array(train_perfs)-np.array(train_perfs0)
    learnings_test = np.array(test_perfs)-np.array(test_perfs0)
    delta_learning = np.mean(learnings_train)-np.mean(learnings_test)
    if means:
        return round(np.mean(learnings_train),4), round(np.mean(learnings_test),4), round(delta_learning,4)
    else:
        return learnings_train, learnings_test, round(delta_learning,4)
        
#pipeline

def train_grid(train,test,tokenizer,epoch_bench,label_dicts,target_dir="target_model",seed=42,compute=True,print_graphs=True,start_at=0,model_name="bert",node=None,site=None,is_hcl=False):
    """trains model and returns metrics"""
    accs = []
    times = []
    datas = []
    mean_energies = []
    total_energies = []
    learnings_train = []
    learnings_test = []
    delta_learnings = []
    
    if compute:
        for i,epochs in enumerate(epoch_bench):
            if i >= start_at:
                print(f"Training epoch size : {epochs}")
                if i==0:
                    if is_hcl:
                        target_model = patho.get_model(model_name,label_dicts)
                    else:
                        target_model = utils.get_model(model_name,label_dicts)
                else:
                    target_model = utils.reload_model(f"{target_dir}/model_{epoch_bench[i-1]}",label_dicts)
                if i==0:
                    todo_epochs=epochs
                else:
                    todo_epochs=epochs-epoch_bench[i-1]
                t1 = perf_counter()
                dt1 = energy.get_time()
                acc = utils.train_model(target_model,f"{target_dir}/model_{epochs}",train,test,seed=seed,num_train_epochs=todo_epochs,is_hcl=is_hcl)
                dt2 = energy.get_time()
                data = energy.get_power(node, site, dt1, dt2, metric="bmc_node_power_watt")
                t2 = perf_counter()
                delay = (t2-t1)/60
                if epochs==epoch_bench[-1]:
                    plot_distrib=True
                else:
                    plot_distrib=False
                learning_train, learning_test, delta_learning = get_learning(target_model,utils.get_model(model_name,label_dicts),train,test,tokenizer,save_dir=f"{target_dir}/model_{epochs}",save=True,redo=True,plot_distrib=plot_distrib,means=True)
                if i==0:
                    datas.append(data)
                    times.append(delay)
                else:
                    if start_at==0:
                        datas.append(datas[i-1]+data)
                    else:
                        datas.append(None)
                    times.append(times[i-1]+delay)
                learnings_train.append(learning_train)
                learnings_test.append(learning_test)
                delta_learnings.append(delta_learning)
                if start_at==0:
                    data = datas[i]
                    mean_energy, total_energy = np.mean(data),np.sum(data)/1000
                else:
                    total_energy = (total_energies[i-1]*1000+np.sum(data))/1000
                    N_i = (1000*total_energies[i-1])/mean_energies[i-1]
                    mean_energy = total_energy*1000/(N_i+len(data))
                accs.append(acc)
                mean_energies.append(mean_energy)
                total_energies.append(total_energy)
                
                np.savetxt(f"{target_dir}/times_{i}.txt",[times[i]])
                np.savetxt(f"{target_dir}/accs_{i}.txt",[accs[i]])
                np.savetxt(f"{target_dir}/mean_energies_{i}.txt",[mean_energies[i]])
                np.savetxt(f"{target_dir}/total_energies_{i}.txt",[total_energies[i]])
                np.savetxt(f"{target_dir}/learnings_train_{i}.txt",[learnings_train[i]])
                try:
                    np.savetxt(f"{target_dir}/learnings_test_{i}.txt",[learnings_test[i]])
                except:
                    print(f"learning test save {i} bug")
                np.savetxt(f"{target_dir}/delta_learnings_{i}.txt",[delta_learnings[i]])
            else:
                times.append(np.loadtxt(f"{target_dir}/times_{i}.txt"))
                accs.append(np.loadtxt(f"{target_dir}/accs_{i}.txt"))
                mean_energies.append(np.loadtxt(f"{target_dir}/mean_energies_{i}.txt"))
                total_energies.append(np.loadtxt(f"{target_dir}/total_energies_{i}.txt"))
                datas.append(None)
                learnings_train.append(np.loadtxt(f"{target_dir}/learnings_train_{i}.txt"))
                learnings_test.append(np.loadtxt(f"{target_dir}/learnings_test_{i}.txt"))
                delta_learnings.append(np.loadtxt(f"{target_dir}/delta_learnings_{i}.txt"))
                print(f"Loaded epoch size : {epochs}")
                
        np.savetxt(f"{target_dir}/times.txt",times)
        np.savetxt(f"{target_dir}/accs.txt",accs)
        np.savetxt(f"{target_dir}/mean_energies.txt",mean_energies)
        np.savetxt(f"{target_dir}/total_energies.txt",total_energies)
        np.savetxt(f"{target_dir}/learnings_train.txt",learnings_train)
        np.savetxt(f"{target_dir}/learnings_test.txt",learnings_test)
        np.savetxt(f"{target_dir}/delta_learnings.txt",delta_learnings)
    else:
        times = np.loadtxt(f"{target_dir}/times.txt")
        accs = np.loadtxt(f"{target_dir}/accs.txt")
        mean_energies = np.loadtxt(f"{target_dir}/mean_energies.txt")
        total_energies = np.loadtxt(f"{target_dir}/total_energies.txt")
        learnings_train = np.loadtxt(f"{target_dir}/learnings_train.txt")
        learnings_test = np.loadtxt(f"{target_dir}/learnings_test.txt")
        delta_learnings = np.loadtxt(f"{target_dir}/delta_learnings.txt")
        a,b,c = get_learning(utils.reload_model(f"{target_dir}/model_{epoch_bench[-1]}",label_dicts),utils.get_model("bert",label_dicts),train,test,tokenizer,
                                save_dir=f"{target_dir}/model_{epoch_bench[-1]}",save=False,redo=False,plot_distrib=print_graphs)
    
    if print_graphs:
        fig,ax = plt.subplots(2,3,figsize=(12,8))
        fig.suptitle("Training phase")
        X = [str(T) for T in epoch_bench]

        ax[0,0].plot(X,times)
        ax[0,0].set_xlabel("Training epochs")
        ax[0,0].set_ylabel("Training time")
        ax[0,1].plot(X,total_energies)
        ax[0,1].set_ylabel("Energy consumption (Total kW)")
        ax[0,1].set_xlabel("Training epochs")
        ax[0,2].plot(X,mean_energies)
        ax[0,2].set_ylabel("Energy consumption (Mean W)")
        ax[0,2].set_xlabel("Training epochs")
        ax[1,0].plot(X,accs)
        ax[1,0].set_ylabel("Eval accuracy")
        ax[1,0].set_xlabel("Training epochs")
        ax[1,1].plot(X,learnings_train,label="train")
        ax[1,1].plot(X,learnings_test,label="test")
        ax[1,1].set_ylabel("Learning curve")
        ax[1,1].set_xlabel("Training epochs")
        ax[1,2].plot(X,delta_learnings)
        ax[1,2].set_ylabel("Overfit curve (learning difference)")
        ax[1,2].set_xlabel("Training epochs")
        ax[1,1].legend()
        fig.tight_layout()