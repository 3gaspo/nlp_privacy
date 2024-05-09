## mia -- Gaspard Berthelier
# for mia applications

## imports
#technical
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, classification_report, roc_curve
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import concatenate_datasets, Value
import torch
import xgboost as xgb
from scipy.stats import gaussian_kde
from sklearn.utils import shuffle

#gaspard
import utils
import dp

#usuals
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


#dataset

def get_shadow_dataset(dataset,tokenizer,target_dir,label_dicts,do_dp=None):
    """returns dataset with generator predictions for shadow training
    (do_dp is either None or "dp" or "vanilla")
    """
    if do_dp:
        target_model = dp.reload_dp_model("bert",target_dir,label_dicts=label_dicts,model_type=do_dp)
    else:
        target_model = utils.reload_model(target_dir,label_dicts)
    if "label" not in dataset.column_names:
        dataset = dataset.rename_column("true_label","label")
    dataset = dataset.map(lambda data: utils.add_predictions(data["text"],data["label"],target_model,tokenizer))
    if "true_label" not in dataset.column_names:
        dataset = dataset.rename_column("label","true_label")
    return dataset

def get_shadow_splits(data,n_splits=1,random_state=41,test_size=0.5):
    """returns index splits for dataset as in/out dicts"""
    rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    train_tests = []
    for train_index, test_index in rs.split(data):
        train_tests.append({"in":train_index,"out":test_index})
    return train_tests

def float_to_label(data,label_name,label_type="int"):
    """turns label into int"""
    if label_type=="int":
        data["label"] = int(data[label_name])
    else:
        data["label"] = float(data[label_name])
    return data
def set_label(dataset,label_name):
    """returns datasets with label_name as label column"""
    if label_name=="regression":
        label_type="float"
    else:
        label_type="int"
    dataset = dataset.map(lambda data: float_to_label(data,label_name,label_type=label_type))
    new_features = dataset.features.copy()
    if label_type=="int":
        new_features["label"] = Value('int64')
        dataset = dataset.cast(new_features)
    return dataset


    
def get_in_out(dataset,real_mems,seed=42,out_size=0.5,include_mem=True):
    """returns in/out with mems inside in"""
    N = dataset.num_rows
    status = []
    for k in range(N):
        if include_mem:
            if k<= int(N*(1-out_size)) or k in real_mems:
                status.append(1)
            else:
                status.append(0)
        else:
            if k<= int(N*(1-out_size)) and k not in real_mems:
                status.append(1)
            else:
                status.append(0)
    dataset = dataset.add_column("status",status)
    return dataset

#shadow training
def compute_metrics_for_regression(eval_pred):
    """returns mse and accuracy for regression"""
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)
    mse = mean_squared_error(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    acc = sum([1 for e in single_squared_errors if e < 0.5]) / len(single_squared_errors)
    return {"acc":acc,"mse":mse}
class RegressionTrainer(Trainer):
    """extends usual trainer for regression"""
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
def train_shadows(sh_dataset,train_tests,output_dir,seed,label_dicts,num_epochs=10,save_dir="shadows",model_name="bert",save=False,start_at=0,is_hcl=False):
    """trains shadows to imitate target model"""
    n_shadows = len(train_tests)
    metrics = [[] for k in range(n_shadows)]
    for k in range(n_shadows):
        if k>=start_at:
            print(f"Training shadow {k}")
            t1 = perf_counter()
            if is_hcl:
                sh_model = patho.get_model(model_name,label_dicts)
            else:
                sh_model = utils.get_model(model_name,label_dicts)
            if len(label_dicts[0].values())==1:
                training_args = TrainingArguments(output_dir=f"{output_dir}/shadow_{k}",seed=seed)
                trainer = RegressionTrainer(
                    compute_metrics = compute_metrics_for_regression,
                    model=sh_model,
                    args=training_args,
                    train_dataset=sh_dataset.select(train_tests[k]["in"]),
                    eval_dataset=sh_dataset.select(train_tests[k]["out"]))
                trainer.train()
                metrics_list = trainer.evaluate()
                metrics[k] = [metrics_list["eval_acc"],metrics_list["eval_mse"]]
            else:
                metrics[k] = utils.train_model(sh_model,f"{output_dir}/shadow_{k}",sh_dataset.select(train_tests[k]["in"]),sh_dataset.select(train_tests[k]["out"]),seed=seed,num_train_epochs=num_epochs,is_hcl=is_hcl)
            utils.get_delay(t1)
            np.savetxt(f"{save_dir}/sh_metrics_{k}",[metrics[k]])
        else:
            metrics[k]=np.loadtxt(f"{save_dir}/sh_metrics_{k}")
    
    np.savetxt(f"{save_dir}/sh_metrics",metrics)
    return metrics


#mia training
def get_mia_dataset(dataset,train_tests,shadow_dir,tokenizer,label_dicts):
    """generates mia matrix of shadow model predictions"""
    X_train = np.array([])
    y_train = np.array([])
    n_shadows = len(train_tests)
    
    for k in range(n_shadows):
        shadow_model = utils.reload_model(f"{shadow_dir}/shadow_{k}",label_dicts)
        dataset_sh = dataset.map(lambda data: utils.add_predictions(data["text"],data["label"],shadow_model,tokenizer))
        predicted_in = dataset_sh.select(train_tests[k]["in"])["regression"]
        predicted_out = dataset_sh.select(train_tests[k]["out"])["regression"]
        X_train = np.concatenate((X_train,np.array(predicted_in), np.array(predicted_out)), axis=0)
        y_train = np.concatenate((y_train,np.array([1 for k in range(len(predicted_in))]),np.array([0 for k in range(len(predicted_out))])),axis=0)
    
    np.savetxt(f'{shadow_dir}/X_train.txt', X_train, delimiter=',')
    np.savetxt(f'{shadow_dir}/y_train.txt', y_train, delimiter=',')

    return X_train, y_train


def get_scores(X_train):
    """returns scores matrix"""
    X = np.abs(X_train)
    return np.array(X)-np.array(X).astype(int)

def get_separate_scores(X_train):
    """returns scores matrix (depending on true value)"""
    return np.array(X_train)-np.array(X_train).astype(int)
    
def get_labels(X_train):
    """returns label matrix, only works with regression"""
    X = np.abs(X_train)
    return np.array(X).astype(int)

def train_mia(X_train,y_train,train_type="simple",scale_pos_weight=1):
    """trains xgb model on x_train"""
    if train_type=="simple":
        xgb_model = xgb.XGBClassifier(max_depth=1,max_leaves=2,scale_pos_weight=scale_pos_weight)
    elif train_type=="double":
        xgb_model = xgb.XGBClassifier(max_depth=2,max_leaves=4,scale_pos_weight=scale_pos_weight)
    else:
        xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)
    X_train = X_train.reshape((-1,1))
    y_train = y_train.reshape((-1,1))
    xgb_model.fit(X_train, y_train)
    return xgb_model

def test_mia(dataset,xgb_model,seed=42,mia_label="separate_score"):
    """returns mia predictions"""
    dataset = dataset.map(lambda data: {"infered_status": xgb_model.predict(data[mia_label])})
    pred = np.array(dataset["infered_status"])
    true = np.array(dataset["status"])
    return pred, true

#analysis
def concatenate_sh_sets(dataset_sh,train_tests):
    """extends dataset_sh columns to x_train size"""
    n_shadows = len(train_tests)
    true_labels = []
    target_label_ids = []
    target_scores = []
    target_regression = []
    target_separate_scores = []
    for k in range(n_shadows):
        n_out = len(train_tests[k]["out"])
        in_dx = [idx for step,idx in enumerate(train_tests[k]["in"]) if step<n_out]
        dataset_in = dataset_sh.select(in_dx)
        dataset_out = dataset_sh.select(train_tests[k]["out"])
        true_labels+=dataset_in["true_label"]
        true_labels+=dataset_out["true_label"]
        target_regression+=dataset_in["regression"]
        target_regression+=dataset_out["regression"]
        target_scores+=dataset_in["prediction_score"]
        target_scores+=dataset_out["prediction_score"]
        target_label_ids+=dataset_in["predicted_label"]
        target_label_ids+=dataset_out["predicted_label"]
        target_separate_scores+=dataset_in["separate_score"]
        target_separate_scores+=dataset_out["separate_score"]

    return true_labels,target_label_ids,target_scores,target_regression,target_separate_scores
        
def compare_list(L1,L2,mode=True):
    """returns idx where L1==L2"""
    n = len(L1)
    where = []
    for k in range(n):
        if (mode and L1[k]==L2[k]) or (not mode and L1[k]!=L2[k]) :
            where.append(k)            
    return where

def find_ins(L1,L2,mode=True):
    """returns idx where L1 in L2 or L2 in L1"""
    n = len(L1)
    where = []
    for k in range(n):
        if (mode and (L1[k] in L2)) or (not mode and (L1[k] not in L2)):
            where.append(k)
    return where


def check_mia(X_train,y_train,dataset_sh,train_tests,sh_label="true_label"):
    """plots graphs for mia analysis"""
    X_train_scores = get_scores(X_train)
    X_train_scores_separate = get_separate_scores(X_train)
    true_labels,target_label_ids,target_scores,target_regression,target_separate_scores = concatenate_sh_sets(dataset_sh,train_tests)

    idx_true = compare_list(dataset_sh["true_label"],dataset_sh["predicted_label"])
    idx_false = compare_list(dataset_sh["true_label"],dataset_sh["predicted_label"],mode=False)
    idx_true_sh = compare_list(true_labels,target_label_ids)
    idx_false_sh = compare_list(true_labels,target_label_ids,mode=False)

    fig,ax = plt.subplots(2,4,figsize=(20,10))
    
    #predicted by target vs true label
    xy0 = np.vstack([dataset_sh["true_label"],dataset_sh["predicted_label"]])
    z0 = gaussian_kde(xy0)(xy0)
    ax[0,0].scatter(dataset_sh["true_label"],dataset_sh["predicted_label"],c=z0)
    ax[0,0].set_title("Predicted label by target model")
    ax[0,0].set_xlabel("True label")
    ax[0,0].set_ylabel("Target label")
    #ax[0,0].legend()
    
    #predicted by shadows vs shadow label
    if sh_label=="regression":
        label_sh = target_regression
    elif sh_label=="predicted_label":
        label_sh = target_label_ids
    elif sh_label=="true_label":
        label_sh = true_labels
    xy0 = np.vstack([label_sh,np.abs(X_train).reshape((1,-1))])
    z0 = gaussian_kde(xy0)(xy0)
    ax[0,1].scatter(label_sh,np.abs(X_train),c=z0) 
    ax[0,1].set_title("Predictions by shadow models")
    ax[0,1].set_xlabel(f"Shadows' label ({sh_label})")
    ax[0,1].set_ylabel("Shadows' prediction (label+score)")
    
    #predicted by shadows vs true status
    xy0 = np.vstack([X_train_scores_separate.reshape((1,-1)),y_train.reshape((1,-1))])
    z0 = gaussian_kde(xy0)(xy0)
    ax[0,2].scatter(X_train_scores_separate,y_train,c=z0)
    ax[0,2].set_title("MIA classification training") 
    ax[0,2].set_xlabel("Shadow's separate score")
    ax[0,2].set_ylabel("Status (in/out of shadows)")
    
    #shadows' score vs true status
    xy0 = np.vstack([X_train_scores.reshape((1,-1)),y_train.reshape((1,-1))])
    z0 = gaussian_kde(xy0)(xy0)
    ax[0,3].scatter(X_train_scores,y_train,c=z0)
    ax[0,3].set_title("MIA classification training")
    ax[0,3].set_xlabel("Shadow's score")
    ax[0,3].set_ylabel("Status (in/out of shadows)")


    #target score vs true label
    xy0 = np.vstack([true_labels,target_separate_scores])
    z0 = gaussian_kde(xy0)(xy0)
    ax[1,0].scatter(true_labels,target_separate_scores,c=z0) 
    ax[1,0].set_title("Target's prediction score")
    ax[1,0].set_xlabel("True label")
    ax[1,0].set_ylabel("Target's separate score")
    ax[1,0].legend()
    
    #target score vs shadows score
    xy0 = np.vstack([target_separate_scores,X_train_scores_separate.reshape((1,-1))])
    z0 = gaussian_kde(xy0)(xy0)
    ax[1,1].scatter(target_separate_scores,X_train_scores_separate,c=z0) 
    ax[1,1].set_title("Prediction scores")
    ax[1,1].set_xlabel("Target's separate score")
    ax[1,1].set_ylabel("Shadows' separate score")
    
    #target regression vs true status
    xy0 = np.vstack([dataset_sh["separate_score"],dataset_sh["status"]])
    z0 = gaussian_kde(xy0)(xy0)
    ax[1,2].scatter(dataset_sh["separate_score"],dataset_sh["status"],c=z0) 
    ax[1,2].set_title("MIA classification objective")
    ax[1,2].set_ylabel("Status (in/out target training)")
    ax[1,2].set_xlabel("Target's separate scores")
    
    #target score vs true status
    xy0 = np.vstack([dataset_sh["prediction_score"],dataset_sh["status"]])
    z0 = gaussian_kde(xy0)(xy0)
    ax[1,3].scatter(dataset_sh["prediction_score"],dataset_sh["status"],c=z0) 
    ax[1,3].set_title("MIA classification objective")
    ax[1,3].set_ylabel("Status (in/out target training)")
    ax[1,3].set_xlabel("Target's score")

    fig.tight_layout()
    plt.show()
    
    
def get_mia_tree(xgb_model):
    """returns xgb tree"""
    xgb.plot_tree(xgb_model,num_trees=xgb_model.get_booster().best_iteration)
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    
def get_simple_threshold(xgb_model):
    """returns simple xgb threshold"""
    t = float(xgb_model.get_booster()[xgb_model.get_booster().best_iteration].get_dump()[0][6:15])
    return t

def draw_roc(X_train, y_train, title, complete=True):
    """plots roc curve for x_train"""
    fig = plt.figure(figsize=(5,5))
    if not complete:
        fpr, tpr, thresholds = roc_curve(y_train, X_train, pos_label=1)
        plt.plot(fpr,tpr)
    else:
        fpr1, tpr1, thresholds = roc_curve(y_train, X_train, pos_label=1)
        fpr2, tpr2, thresholds = roc_curve(y_train, get_separate_scores(X_train), pos_label=1)
        fpr3, tpr3, thresholds = roc_curve(y_train, get_scores(X_train), pos_label=1)
        plt.plot(fpr1,tpr1)
        plt.plot(fpr2,tpr2)
        plt.plot(fpr3,tpr3)
        plt.legend(["regression","separate scores","scores"])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve for {title}")
    
def mia_results(xgb_model,tokenizer,dataset_sh,seed,mia_label="separate_score",print_report=True,print_graphs=True):
    """returns mia results"""
    pred,true = test_mia(dataset_sh,xgb_model,seed=seed,mia_label=mia_label)
    pred = np.array(pred).reshape((1,-1))[0]
    acc = classification_report(true,pred,output_dict=True)["accuracy"]
    if print_report:
        print("-----------------------")
        print(classification_report(true,pred))
        print("-----------------------")
        n_pred_ins = len(np.where(pred==1)[0])
        print(f"% predicted in : {round(100*n_pred_ins/len(true))} % (target:50%)")
    if print_graphs:
        plt.scatter(dataset_sh[mia_label],np.array(pred)+0.01)
        plt.scatter(dataset_sh[mia_label],true)
        plt.xlabel("MIA input")
        plt.ylabel("status (in/out)")
        plt.legend(["predicted status","true status"])
        plt.title("MIA decision boundaries")
        plt.show()
        get_mia_tree(xgb_model)
    return round(acc,4)


#pipeline

def mia_pipeline(tokenizer,dataset,seed,label_dicts,real_mems,sh_label_name="true_label",mia_label="separate_score",n_shadows=1,num_epochs=10,train_type="simple",pos_weight=1,do_shadows_training=True,shadow_dir="mia",model_name="bert",build_mia=True,print_graphs=True,print_report=True):
    """pipeline for mia on classification score """
    dataset = set_label(dataset,sh_label_name)
    train_tests = get_shadow_splits(dataset,n_splits=n_shadows,random_state=seed-1,test_size=0.5)
    
    if do_shadows_training:
        t1 = perf_counter()
        print("Training shadows")
        metrics = train_shadows(dataset,train_tests,shadow_dir,seed,label_dicts,num_epochs=num_epochs,
                                save=True,save_dir=shadow_dir,model_name=model_name,start_at=0)
        utils.get_delay(t1)
    else:
        metrics = np.loadtxt(f"{shadow_dir}/sh_metrics")
    if print_report:
        if sh_label_name=="regression":
            print(f"Shadow metrics (acc,mse) : ",metrics)
        else:
            print(f"Shadow acc ",metrics)

    if build_mia:
        t1 = perf_counter()
        print("Building MIA dataset")
        X_train, y_train = get_mia_dataset(dataset,train_tests,shadow_dir,tokenizer,label_dicts)
        utils.get_delay(t1)
    else:
        X_train = np.loadtxt(f'{shadow_dir}/X_train.txt',delimiter=',').reshape((-1,1))
        y_train = np.loadtxt(f'{shadow_dir}/y_train.txt',delimiter=',').reshape((-1,1))
    
    if print_graphs:
        complete = True
        if sh_label_name=="regression":
            complete = False
        draw_roc(X_train, y_train,"shadow",complete=True)
        draw_roc(dataset["regression"], dataset["status"],"target data",complete=True)
    
    X = {"prediction_score":get_scores(X_train),"separate_score":get_separate_scores(X_train),"regression":X_train}
    xgb_model = train_mia(X[mia_label],y_train,train_type=train_type,scale_pos_weight=pos_weight)
    
    if print_graphs:
        check_mia(X_train,y_train,dataset,train_tests,sh_label=sh_label_name)
        
    acc = mia_results(xgb_model,tokenizer,dataset,seed,mia_label=mia_label,
                                   print_report=print_report,print_graphs=print_graphs)
    
    X_train = [X[mia_label][k] for k in real_mems]
    y_train = [int(y_train[k][0]) for k in real_mems]
    pred = xgb_model.predict(np.array(X_train)).reshape((1,-1))[0]
    mem_acc = len(np.where(pred==np.array(y_train))[0])/len(pred)
    print("Accuracy on counterfactuals : ", round(mem_acc,4))
    
    return train_tests, xgb_model, X, y_train, dataset, acc, mem_acc


    

def draw_separability(dataset,tokenizer,targets_dir,benches,label_dicts,mia_label):
    """draws score distributions during all epochs"""
    X = {}
    for i in range(len(benches)):
        print(f"rebuilding dataset {i}")
        target_dir = f"{targets_dir}/model_{benches[i]}"
        temp_dataset = get_shadow_dataset(dataset,tokenizer,target_dir,label_dicts)
        X[f"model_{benches[i]}"] = temp_dataset[mia_label]
    
    fig = plt.figure()
    print("All done")
    for i in range(len(benches)):
        xy0 = np.vstack([X[f"model_{benches[i]}"],temp_dataset["status"]])
        z0 = gaussian_kde(xy0)(xy0)
        plt.scatter(X[f"model_{benches[i]}"],temp_dataset["status"],c=z0) 
        plt.title(f"MIA classification objective ({benches[i]} epochs)")
        plt.ylabel("Status (in/out target training)")
        plt.xlabel("Target's input")
        plt.pause(0.01)
        
    plt.show()
    
    
def draw_last_separability(mia_dir,dataset):
    """draws zoomed score distributions for last epoch"""
    
    X_train = np.loadtxt(f"{mia_dir}/X_train.txt")
    y_train = np.loadtxt(f"{mia_dir}/y_train.txt")
    X = get_scores(X_train)

    print("")
    print("Shadows")
    plt.scatter(X,y_train)
    plt.xlim([0.99,1])
    plt.show()

    xy0 = np.vstack([X,y_train])
    z0 = gaussian_kde(xy0)(xy0)
    plt.scatter(X,y_train,c=z0)
    plt.xlim([0.998,1])
    plt.show()
    
    print("Tresholds : MIA accuracy")
    for threshold in [0.996,0.9965,0.997,0.9975,0.998,0.9985,0.9986,0.9987,0.999,0.9991,0.9992,0.9993,0.9994]:
        pred = []
        for x in X:
            if x>=threshold:
                pred.append(1)
            else:
                pred.append(0)
        idx_true = np.where(np.array(pred)==np.array(y_train).astype(int))[0]
        print(threshold," : ",round(len(idx_true)/len(y_train),2))
    
    print("")
    print("Target")
    xy0 = np.vstack([dataset["prediction_score"],dataset["status"]])
    z0 = gaussian_kde(xy0)(xy0)
    plt.scatter(dataset["prediction_score"],dataset["status"],c=z0)
    plt.xlim([0.998,1])
    plt.show()

    xy0 = np.vstack([dataset["prediction_score"],dataset["status"]])
    z0 = gaussian_kde(xy0)(xy0)
    plt.scatter(dataset["prediction_score"],dataset["status"],c=z0)
    plt.xlim([0.9995,1])
    
    print("Tresholds : MIA accuracy")
    for threshold in [0.999,0.9995,0.9996,0.99965,0.9997,0.99975,0.9998,0.99985,0.9999,0.99991,0.99995,0.99999]:
        pred = []
        for x in dataset["prediction_score"]:
            if x>=threshold:
                pred.append(1)
            else:
                pred.append(0)
        idx_true = np.where(np.array(pred)==np.array(dataset["status"]).astype(int))[0]
        print(threshold," : ",round(len(idx_true)/len(dataset["status"]),2))
        
def shuffle_mia(dataset,seed=42,test_size=0.1,mia_label="regression"):
    """shuffles training dataset for mia and returns train test"""
    
    X = np.array(dataset[mia_label])
    y = np.array(dataset["status"])
    X, y = shuffle(X, y, random_state=seed)

    n = X.shape[0]
    X_0 = X[0:int((1-test_size)*n)]
    X_1 = X[int((1-test_size)*n):]
    y_0 = y[0:int((1-test_size)*n)]
    y_1 = y[int((1-test_size)*n):]
    return X_0,y_0,X_1,y_1

def target_mia(target_dir,dataset,label_dicts,tokenizer,seed=42,redo_dataset=True,mia_label="regression",pos_weight=1,debug=False):
    """do mia on target model data directly"""
    if mia_label=="regression":
        train_type="multi"
    else:
        train_type="simple"
        
    if redo_dataset:
        dataset = get_shadow_dataset(dataset,tokenizer,target_dir,label_dicts)
    X_0,y_0,X_1,y_1 = shuffle_mia(dataset,seed=seed,test_size=0.1,mia_label=mia_label)

    xgb_model = train_mia(X_0,y_0,train_type=train_type,scale_pos_weight=pos_weight) #tried other scale, best : 0.9
    pred = xgb_model.predict(np.array(X_1)).reshape((1,-1))[0]
    true = np.array(y_1).reshape((1,-1))[0]
    acc = classification_report(true,pred,output_dict=True)["accuracy"]
    if debug:
        return round(acc,4), pred, true
    else:
        return round(acc,4)
    
    
def dp_mia(dataset,tokenizer_dp,label_dicts,seed=42,pos_weight=1,mia_label="regression",test_size=0.1,debug=False):
    """mia pîpeline to attack dp model"""
    
    if mia_label=="regression":
        train_type="multi"
    else:
        train_type="simple"

    print("-------")
    print("Vanilla model")
    dataset = get_shadow_dataset(dataset,tokenizer_dp,"dp_mia",label_dicts,do_dp="vanilla")
    X_0,y_0,X_1,y_1 = shuffle_mia(dataset,seed=seed,test_size=test_size,mia_label=mia_label)
    print("MIA train size : ",X_0.shape[0])
    print("MIA eval size : ",X_1.shape[0])
    xgb_model = train_mia(np.array(X_0).reshape((-1,1)),np.array(y_0).reshape((-1,1)),train_type=train_type,scale_pos_weight=pos_weight)
    pred = xgb_model.predict(X_1.reshape((-1,1))).reshape((1,-1))[0] 
    print("Vanilla MIA accuracy: ",round(len(np.where(np.array(pred)==np.array(y_1).reshape((1,-1))[0])[0])/len(pred),4))
    if debug:
        return pred, y_1
    
    print("")
    print("-------")
    print("DP model")
    dataset = get_shadow_dataset(dataset,tokenizer_dp,"dp_mia",label_dicts,do_dp="dp")
    X_0,y_0,X_1,y_1 = shuffle_mia(dataset,seed=seed,test_size=test_size,mia_label=mia_label)
    print("MIA train size : ",X_0.shape[0])
    print("MIA eval size : ",X_1.shape[0])
    xgb_model = train_mia(np.array(X_0).reshape((-1,1)),np.array(y_0).reshape((-1,1)),train_type=train_type,scale_pos_weight=pos_weight)
    pred = xgb_model.predict(np.array(X_1).reshape((-1,1))).reshape((1,-1))[0]
    print("DP MIA accuracy: ",round(len(np.where(np.array(pred)==np.array(y_1).reshape((1,-1))[0])[0])/len(pred),4))

    
def fed_mia(dataset,dataset_list,tokenizer_dp,label_dicts,pos_weight=1,mia_label="regression",seed=42,test_size=0.1):
    """mia pîpeline to attack fed model"""

    if mia_label=="regression":
        train_type="multi"
    else:
        train_type="simple"
        
    print("Building datasets")
    n_part = len(dataset_list)
    mia_dataset_list = []
    for i in range(n_part):
        n_train, n_test = len(dataset_list[i]["train"]), len(dataset_list[i]["test"])
        n_dp = min(n_train,n_test)
        train_dp = dataset_list[i]["train"].select(range(n_dp)).map(lambda data: {"status":1})
        test_dp = dataset_list[i]["test"].select(range(n_dp)).map(lambda data: {"status":0})
        mia_dataset_list.append(concatenate_datasets([train_dp,test_dp]))
        train_dp = dp.get_dp_datasets(train_dp,tokenizer_dp,test_size=0.1,seed=seed,split=None)
        test_dp = dp.get_dp_datasets(test_dp,tokenizer_dp,test_size=0.1,seed=seed,split=None)
        print(f"Node {i} MIA 'in' size ",len(train_dp))

    whole_fed_dataset = concatenate_datasets(mia_dataset_list)
    whole_train_test = dataset.train_test_split(test_size=0.1,seed=seed)
    n_train, n_test = len(whole_train_test["train"]), len(whole_train_test["test"])
    n_dp = min(n_train,n_test)
    train_dp = whole_train_test["train"].select(range(n_dp)).map(lambda data: {"status":1})
    test_dp = whole_train_test["test"].select(range(n_dp)).map(lambda data: {"status":0})
    whole_vanilla_dataset = concatenate_datasets([train_dp,test_dp])

    print("")
    print("-------------")
    print("Global vanilla")
    mia_dataset = get_shadow_dataset(whole_vanilla_dataset,tokenizer_dp,f"fed",label_dicts,do_dp="vanilla")
    X_0,y_0,X_1,y_1 = shuffle_mia(mia_dataset,seed=seed,test_size=test_size,mia_label=mia_label)
    print("MIA train size : ",X_0.shape[0])
    print("MIA eval size : ",X_1.shape[0])
    xgb_model = train_mia(np.array(X_0).reshape((-1,1)),np.array(y_0).reshape((-1,1)),train_type=train_type,scale_pos_weight=pos_weight)
    pred = xgb_model.predict(np.array(X_1).reshape((-1,1))).reshape((1,-1))[0]
    acc = round(len(np.where(np.array(pred)==np.array(y_1).reshape((1,-1))[0])[0])/len(pred),4)
    if not acc > 0.53:
        return "too low acc"
    print("Vanilla MIA accuracy: ",acc)
    
    print("")
    print("-------------")
    print("Local vanillas")
    for i in range(n_part):
        print(f"model {i}")
        mia_dataset_list[i] = get_shadow_dataset(mia_dataset_list[i],tokenizer_dp,f"fed/model_{i}",label_dicts,do_dp="vanilla")
        #X_0,y_0,X_1,y_1 = shuffle_mia(mia_dataset_list[i],seed=seed,test_size=test_size,mia_label=mia_label)
        #xgb_model = train_mia(np.array(X_0).reshape((-1,1)),np.array(y_0).reshape((-1,1)),train_type=train_type,scale_pos_weight=pos_weight)
        #pred = xgb_model.predict(np.array(X_1).reshape((-1,1))).reshape((1,-1))[0]
        print("MIA eval size : ",mia_dataset_list[i].num_rows)
        pred = xgb_model.predict(np.array(mia_dataset_list[i][mia_label]).reshape((-1,1))).reshape((1,-1))[0]
        print("Vanilla MIA accuracy: ",round(len(np.where(np.array(pred)==np.array(mia_dataset_list[i]["status"]).reshape((1,-1))[0])[0])/len(pred),4))



    print("")
    print("-------------")
    print("Fed model")
    mia_dataset = get_shadow_dataset(whole_fed_dataset,tokenizer_dp,"fed",label_dicts,do_dp="fed")
    #X_0,y_0,X_1,y_1 = shuffle_mia(mia_dataset,seed=seed,test_size=test_size,mia_label=mia_label)
    #xgb_model = train_mia(np.array(X_0).reshape((-1,1)),np.array(y_0).reshape((-1,1)),train_type=train_type,scale_pos_weight=pos_weight)
    #pred = xgb_model.predict(np.array(X_1).reshape((-1,1))).reshape((1,-1))[0]
    pred = xgb_model.predict(np.array(mia_dataset[mia_label]).reshape((-1,1))).reshape((1,-1))[0]
    print("MIA eval size : ",mia_dataset.num_rows)
    print("Fed MIA accuracy: ",round(len(np.where(np.array(pred)==np.array(mia_dataset["status"]).reshape((1,-1))[0])[0])/len(pred),4))