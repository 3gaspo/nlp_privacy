## extr -- Gaspard Berthelier
#extractibility applications

## imports
#transformers
from transformers import AutoTokenizer,pipeline,Trainer,TrainingArguments
from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM
import re

#gaspard
import utils

#usuals
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter


## dataset
def get_text_gen_tokenizer(config="distilgpt2",is_hcl=False):
    """returns tokenizer for text generation"""
    if is_hcl:
        tokenizer = AutoTokenizer.from_pretrained("models/distilgpt2")
    else:
        tokenizer = AutoTokenizer.from_pretrained(config)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def group_texts(examples,chunk_size=150):
    """function for contatenating texts"""
    concatenated_examples = {k: sum(examples[k],[]) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total__length = (total_length // chunk_size) * chunk_size
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k,t in concatenated_examples.items()}
    return result

def tokenize_fct(tokenizer,element,context_length,remove_small=True):
    """adds tokenized column"""
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length or (not remove_small and length>20):
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

def get_texts(tokenizer,list_ids):
    """decodes texts"""
    texts = []
    for list_id in list_ids:
        decoded = tokenizer.decode(list_id)
        texts.append(decoded)
    return "".join(texts)

def get_text_gen_dataset(dataset,tokenizer,context_length=150,is_hcl=False):
    """returns tokenized dataset for text generation"""
    if is_hcl:
        tokenized_dataset = dataset.map(lambda data: tokenize_fct(tokenizer,data,context_length,remove_small=False), batched=True, remove_columns=dataset.column_names).map(lambda data: group_texts(data,chunk_size=context_length),batched=True)
    else:
        tokenized_dataset = dataset.map(lambda data: tokenize_fct(tokenizer,data,context_length), batched=True, remove_columns=dataset.column_names)
    
    tokenized_dataset = tokenized_dataset.map(lambda data: {"text":get_texts(tokenizer,data["input_ids"])}) #to have texts the same size as input ids
    return tokenized_dataset

def plot_gen_dataset(dataset_gen,context_length=150):
    """plots dataset prompt distribution"""
    fig, axes = plt.subplots(1, 4,figsize=(15, 4))
    words = utils.get_word_count(dataset_gen["text"])
    uniques = utils.get_unique_count(dataset_gen["text"])
    tokens = utils.get_num_tokens(dataset_gen["input_ids"])
    unique_tokens = utils.get_unique_tokens(dataset_gen["input_ids"])
    
    sns.histplot(ax = axes[0],x=words).set(title="Number of words distribution")
    sns.histplot(ax = axes[1],x=uniques).set(title="Number of unique words distribution")
    sns.histplot(ax = axes[2],x=tokens).set(title="Number of tokens distribution")
    axes[2].set_xlim([min(0,context_length-50),context_length+50])
    sns.histplot(ax = axes[3],x=unique_tokens).set(title="Number of unique tokens distribution")
    fig.tight_layout()
    
    print("Mean words : ",round(np.mean(words),1))
    print("Mean unique words : ",round(np.mean(uniques),1))
    print("Mean tokens : ",round(np.mean(tokens),1))
    print("Mean unique tokens : ",round(np.mean(unique_tokens),1))

def get_texts_info(texts,redo=True,save=True,save_dir="extr",d_object=False):
    """returns text distribution values"""
    sizes, uniques, words = [],[],[]
    mean_sizes, mean_uniques, mean_words = [],[],[]
    
    n = len(texts)
    if redo:
        for k in range(n):
            sizes.append(utils.get_size_count(texts[k]))
            uniques.append(utils.get_unique_count(texts[k]))
            words.append(utils.get_word_count(texts[k]))
        for k in range(n):
            mean_sizes.append(round(np.mean(sizes[k]),1))
            mean_uniques.append(round(np.mean(uniques[k]),1))
            mean_words.append(round(np.mean(words[k]),1))
    
        if save==True:
            if d_object:
                for k in range(n):
                    np.savetxt(f"{save_dir}/sizes_{k}.txt",sizes[k])
                    np.savetxt(f"{save_dir}/words_{k}.txt",uniques[k])
                    np.savetxt(f"{save_dir}/uniques_{k}.txt",words[k])
            else:
                np.savetxt(f"{save_dir}/sizes.txt",sizes)
                np.savetxt(f"{save_dir}/words.txt",uniques)
                np.savetxt(f"{save_dir}/uniques.txt",words)
    else:
        if d_object:
            for k in range(n):
                sizes.append(np.loadtxt(f"{save_dir}/sizes_{k}.txt").astype(int))
                words.append(np.loadtxt(f"{save_dir}/words_{k}.txt").astype(int))
                uniques.append(np.loadtxt(f"{save_dir}/uniques_{k}.txt").astype(int))
        else:
            sizes = np.loadtxt(f"{save_dir}/sizes.txt").astype(int)
            words = np.loadtxt(f"{save_dir}/words.txt").astype(int)
            uniques = np.loadtxt(f"{save_dir}/uniques.txt").astype(int)            
        
        for k in range(n):
            mean_sizes.append(round(np.mean(sizes[k]),1))
            mean_uniques.append(round(np.mean(uniques[k]),1))
            mean_words.append(round(np.mean(words[k]),1))
            
    return sizes, uniques, words, mean_sizes, mean_uniques, mean_words


def find_idx(original_dataset,new_dataset,idx,step):
    """returns indexes of idx in nex_dataset"""
    if step<=15:
        text = original_dataset["text"][idx]
        size = len(text)
        new_size = int(size/step)
        texts = []
        new_idx = 0
        for k in range(step):
            if k<step-1:
                texts.append(text[new_idx:new_idx+new_size])
            else:
                texts.append(text[new_idx:size])
            new_idx = new_idx+new_size

        found = False
        i = 0

        for try_text in texts:
            for j,new_text in enumerate(new_dataset["text"]):
                if try_text in new_text:
                    i=j
                    found=True
                    break
            if found:
                break
        if found:
            return i
        else:
            return find_idx(original_dataset,new_dataset,idx,step+1)
    else:
        print("Not found after 15 its...")
        return None
def text_gen_mems(original_dataset,new_dataset,real_mems):
    """returns possible memorized texts"""
    real_mems_gen_temp = [find_idx(original_dataset,new_dataset,idx,1) for idx in real_mems]
    real_mems_gen = []
    for idx in real_mems_gen_temp:
        real_mems_gen += [idx,idx+1,idx+2]
        if idx>0:
            real_mems_gen.append(idx-1)
        if idx>1:
            real_mems_gen.append(idx-2)
    return real_mems_gen
    
    
    
#model
def get_text_gen_model(config="distilgpt2",is_hcl=False):
    """returns model for text generation"""
    if is_hcl:
        return AutoModelForCausalLM.from_pretrained("models/distilgpt2")
    else:
        return AutoModelForCausalLM.from_pretrained(config)
def reload_text_gen_model(name):
    """returns model for text generation"""
    return AutoModelForCausalLM.from_pretrained(name)
def get_text_gen_generator(model,tokenizer,max_new_tokens=15):
    """returns text generator"""
    return pipeline("text-generation", model=model, tokenizer=tokenizer,pad_token_id=50256, num_return_sequences=1, max_new_tokens=max_new_tokens,return_full_text=False)


#training
def train_text_gen(model,tokenizer,train_ds,test_ds,output_dir,seed=42,num_train_epochs=3):
    """trains model for text generation and saves in output_dir"""
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    training_args = TrainingArguments(output_dir=output_dir,seed=seed,num_train_epochs=num_train_epochs)
    trainer = Trainer(model=model,args=training_args,train_dataset=train_ds,eval_dataset=test_ds,data_collator=data_collator)
    trainer.train()
    model.save_pretrained(output_dir)


# extractability

def get_bench(bench_sizes,texts):
    """returns bench sizes for prompts fitted to texts sizes"""
    N_gen = len(texts)
    min_length = np.min([len(texts[k]) for k in range(N_gen)])
    bench = [max(int(5*(1+bench_sizes[i])),round(int(bench_sizes[i]*min_length),-1)) for i in range(len(bench_sizes))]
    return min_length,bench

def get_prompts(benches,dataset):
    """returns sequences of sizes written in benches, randomly extracted from dataset texts"""
    extract_data = [[] for k in range(len(benches))]
    for sequence in dataset["text"]:
        size = len(sequence)
        for i in range(len(benches)):
            start = np.random.randint(0,max(1,size-2*benches[-1]))
            extract_data[i].append(sequence[start:min(size,start+benches[i])])
    return extract_data

def get_repetitions(prompts,original_data,generator,min_size=20,min_words=5,save=False,save_dir="reps/outputs",start_at=0,device=None):
    """returns output prompts, and repetitions when output is in original data"""
    n = len(prompts)
    repetitions = [[] for k in range(n)]
    outputs = [[] for k in range(n)]

    for i in range(n):
        if i>=start_at:
            print(f"Computing repetitions {i}")
            for j,list_prompt in enumerate(tqdm(prompts[i])):
                prompt = "".join(list_prompt)
                output = generator(prompt)[0]["generated_text"].lower()
                outputs[i].append(output)
                size = len(output)
                word_count = len(output.split(" "))

                if size > min_size and word_count > min_words:
                    for k, list_text in enumerate(original_data["text"]):
                        text = "".join(list_text).lower()
                        if output in text:
                            repetitions[i].append(f"{i}[SEP]{j}[SEP]{k}[SEP]{prompt}[SEP]{output}[SEP]{text}")
                        elif size >= 3*min_size:
                            sep = int(size/3)
                            phrases = [output[0:sep],output[sep:2*sep],output[2*sep:size]]
                            for phrase in phrases:
                                if len(phrase.split(" ")) >= 5 and phrase in text:
                                    repetitions[i].append(f"{i}[SEP]{j}[SEP]{k}[SEP]{prompt}[SEP]{phrase}[SEP]{text}")
            if save:
                np.savetxt(f"{save_dir}/repetitions_{i}",repetitions[i],newline="[END]",fmt="%s")
                np.savetxt(f"{save_dir}/outputs_{i}",outputs[i],newline="[END]",fmt="%s")
            print("Done")
        else:
            with open(f"{save_dir}/repetitions_{i}","r") as a:
                    lines = a.readlines()                
            repetitions[i]=reconstruct_lines(lines)
            with open(f"{save_dir}/outputs_{i}","r") as a:
                    lines = a.readlines()                
            outputs[i]=lines
            print(f"{save_dir}/loaded data {i}")
                        
    return repetitions,outputs


#analysis
def get_repartition(repetitions):
    """counts repetitions for each bench"""
    return [len(repetitions[i]) for i in range(len(repetitions))]

def get_total_repetitions(repartitions):
    """counts total repetitions"""
    return sum(repartitions)

def get_unique_reps(basic_reps):
    """returns unique reps for 1 bench""" 
    reps = []
    reps_idx = []
    for rep in basic_reps:
        info = rep.split("[SEP]")
        j = info[1]
        k = info[2]
        if [j,k] not in reps_idx:
            reps_idx.append([j,k])
            reps.append(rep)
    return reps

def explain_rep(rep):
    """explains 1 repetition entry"""
    info = rep.split("[SEP]")
    prompt = info[3]
    output = info[4]
    from_text = info[5]
    prompt_id = info[1]
    bench_id = info[0]
    repeat_id = info[2]
    print("------------")
    print(f"From bench {bench_id}, prompt id {prompt_id}:")
    print(f"{prompt}")
    print("------------")
    print("Output:")
    print(output)
    output = output.replace("(","\(")
    highlight = re.search(output,from_text).span()
    print("------------")
    print("Repeated text: ")
    print(from_text[:highlight[0]]+ f"\x1b[31m{from_text[highlight[0]:highlight[1]]}\x1b[0m" + from_text[highlight[1]:])

    
def get_rep_outputs(L,outputs_num=4):
    """returns list of outputs from rep list for 1 bench
    outputs_num for which info to extract (see reps definition)
    """
    outputs = []
    for rep in L:
        infos = rep.split("[SEP]")
        outputs.append(infos[outputs_num])
    return outputs

def get_rep_idx(L):
    """returns list of indexes from rep list for 1 bench"""
    idx = []
    for rep in L:
        idx.append(int(rep.split("[SEP]")[1]))
    return idx

def reconstruct_lines(lines,endline="[END]"):
    """for reconstrunction of concatenated sentences"""
    L = []
    temp_L = []
    for text in lines:
        if endline in text:
            subtexts = text.split(endline)
            n = len(subtexts)
            temp_L.append(subtexts[0])
            L.append(" ".join(temp_L))
            temp_L = []
            for step in range(1,n-1):
                L.append(subtexts[step])
            temp_L = [subtexts[1]]
        else:
            temp_L.append(text)
    return L

def plot_texts(sizes,uniques,words,mean_sizes,mean_uniques,mean_words,title):
    """plots sizes distributions of texts (multi bench)"""
    n = len(sizes)
    fig, axes = plt.subplots(1, 3,figsize=(15, 4))
    sns.kdeplot(ax = axes[0],data=list(words)).set(title=title)
    sns.kdeplot(ax = axes[1],data=list(uniques)).set(title=title)
    if len(np.unique(sizes[0]))==1:
        sns.histplot(ax = axes[2],data=list(sizes)).set(title=title)
    else:
        sns.kdeplot(ax = axes[2],data=list(sizes)).set(title=title)

    axes[0].set_xlabel("Words")
    axes[0].set_ylabel("Counts")
    axes[1].set_xlabel("Unique words")
    axes[1].set_ylabel("Counts")
    axes[2].set_xlabel("Characters")
    axes[2].set_ylabel("Counts")

    fig.tight_layout()
    plt.plot()
    print("")
    print(title)
    print("Mean words: ",mean_words)
    print("Mean unique words: ",mean_uniques)
    print("Mean sizes: ",mean_sizes)

def plot_words_double(words1,words2,title):
    """plots sizes distributions of texts (multi bench)"""
    plt.figure()
    sns.kdeplot(data=list(words1))
    sns.kdeplot(data=list(words2),linestyle='dashed')
    plt.xlabel("Words")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.legend()
    plt.title(title)
    plt.plot()

def plot_extr_results(repartition,title,bench,N_gen):
    fig = plt.figure(figsize=(4,3))
    m = min(repartition)
    M = max(repartition)
    plt.bar([str(key) for key in bench],repartition)
    plt.title(title)
    plt.xlabel("Prompt size")
    plt.ylabel("Repetition count")
    plt.ylim(m/2,M*2)
    plt.plot()
    print("Repartition : ",repartition)
    print("Total repetitions : ",get_total_repetitions(repartition))

    print("")
    print("Percentage of extracted sequences")
    print("-----------------------------")
    for k in range(len(repartition)):
        print(f"Prompt size of {bench[k]} : {round(100*repartition[k]/N_gen,1)} % extracted")
        
def extr_results(repetitions,bench,N_gen,title,print_graphs=True):
    """plots results for extraction"""
    repartition = get_repartition(repetitions)
    if print_graphs:
        if np.sum(repartition)==0:
            print("Nothing found.")
        else:
            plot_extr_results(repartition,title,bench,N_gen)
    return repartition
        

#pipeline

def extr_pipeline(dataset,real_mems,context_length=150,seed=42,do_training=True,max_new_tokens=20,do_prompts=True,bench_sizes=[0.1,0.25,0.5,0.75],min_size=25,min_words=7,save_dir="extr",print_graphs=True,is_hcl=False,device=None):
    """pipeline for testing extraction on text generation model"""
    tokenizer_gen = get_text_gen_tokenizer(is_hcl=is_hcl)
    dataset_gen = get_text_gen_dataset(dataset,tokenizer_gen,context_length=context_length,is_hcl=is_hcl)
    print("New dataset size: ",dataset_gen.num_rows)
    n = len(bench_sizes)
    
    #dataset distribution
    if print_graphs:
        print("Dataset distribution")
        plot_gen_dataset(dataset_gen,context_length)
        words_texts = utils.get_word_count(dataset_gen["text"])
    real_mems_gen = text_gen_mems(dataset,dataset_gen,real_mems)
    train_gen, test_gen = utils.get_train_test(dataset_gen,real_mems_gen,test_size=0.1)
    if print_graphs:
        print("Potentiel counterfactuals : ",len(real_mems_gen))
        print("Training size : ",train_gen.num_rows)
    
    #training
    if do_training:
        print("Starting text gen training")
        model_gen = get_text_gen_model(is_hcl=is_hcl)
        t1 = perf_counter()
        train_text_gen(model_gen,tokenizer_gen,train_gen,test_gen,save_dir,seed=seed,num_train_epochs=10)
        utils.get_delay(t1)
    else:
        model_gen = reload_text_gen_model(f"{save_dir}")
    generator_gen = get_text_gen_generator(model_gen,tokenizer_gen,max_new_tokens=max_new_tokens)
    
    #generating prompts
    N_gen = len(train_gen["text"])
    if do_prompts:
        min_length,bench = get_bench(bench_sizes,train_gen["text"])
        print("Prompt bench : ", bench)
        print("Generating prompts")
        prompts = get_prompts(bench,train_gen)
        np.savetxt(f"{save_dir}/min_length.txt",[min_length])
        np.savetxt(f"{save_dir}/bench.txt",bench)
    else:
        prompts = [None for k in range(n)]
        min_length = int(np.loadtxt(f"{save_dir}/min_length.txt"))
        bench = np.loadtxt(f"{save_dir}/bench.txt").astype(int)
        print("Min length: ",min_length)
        print("Prompt sizes : ",bench)
    
    #prompt analysis
    sizes, uniques, words_prompt, mean_sizes, mean_uniques, mean_words = get_texts_info(prompts,redo=do_prompts,save=True,save_dir=f"{save_dir}/prompts")
    if print_graphs:
        print("")
        print("Prompts:")
        plot_texts(sizes, uniques, words_prompt, mean_sizes, mean_uniques, mean_words,"Sampled prompts distribution")
    
    #extraction
    if do_prompts:
        t1 = perf_counter()
        print("Starting extraction")
        basic_reps,outputs = get_repetitions(prompts,train_gen,generator_gen,min_size=min_size,min_words=min_words,save=True,save_dir=f"{save_dir}/outputs",start_at=0,device=device)
        utils.get_delay(t1)
    else:
        basic_reps = [[] for k in range(n)]
        outputs = [[] for k in range(n)]
        for i in range(n):
            with open(f"{save_dir}/outputs/outputs_{i}","r") as a:
                    lines = a.readlines()                
            outputs[i]=reconstruct_lines(lines,endline="[END]")
            with open(f"{save_dir}/outputs/repetitions_{i}","r") as a:
                lines = a.readlines()                
            basic_reps[i]=reconstruct_lines(lines,endline="[END]")
    reps = [get_unique_reps(basic_reps[k]) for k in range(n)]
    try:
        print("")
        print("Outputs:")
        sizes, uniques, words, mean_sizes, mean_uniques, mean_words = get_texts_info(outputs,save=True,save_dir=f"{save_dir}/outputs")
    except:
        print('pb for outputs')
        return outputs
    if print_graphs:
        plot_texts(sizes, uniques, words, mean_sizes, mean_uniques, mean_words,"Outputs distribution")
    
    #analysis
    rep_outputs = [get_rep_outputs(reps[k]) for k in range(n)]
    try:
        print("")
        print("Extracted:")
        sizes, uniques, words, mean_sizes, mean_uniques, mean_words = get_texts_info(rep_outputs,save=True,save_dir=f"{save_dir}/reps",d_object=True)
    except:
        print('pb for rep outputs')
        return rep_outputs
    if print_graphs:
        plot_texts(sizes, uniques, words, mean_sizes, mean_uniques, mean_words,"Extracted phrases distribution")
    
    try:
        rep_prompts = [get_rep_outputs(reps[k],3) for k in range(n)]
        rep_texts = [get_rep_outputs(reps[k],5) for k in range(n)]
        sizes, uniques, words_extr_prompts, mean_sizes, mean_uniques, mean_words = get_texts_info(rep_prompts,save=False,save_dir=f"{save_dir}/reps",d_object=True)
        if print_graphs:
            plot_texts(sizes, uniques, words_extr_prompts, mean_sizes, mean_uniques, mean_words,"Extracted prompts distribution")
        sizes, uniques, words_extr_texts, mean_sizes, mean_uniques, mean_words = get_texts_info(rep_texts,save=False,save_dir=f"{save_dir}/reps",d_object=True)
        if print_graphs:
            plot_texts(sizes, uniques, words_extr_texts, mean_sizes, mean_uniques, mean_words,"Extracted texts distribution")
    except:
        print("pb with try")
        
    repartition_train = extr_results(reps,bench_sizes,N_gen,"Extracted in training dataset",print_graphs)
    
    best_prompt_bench = np.argmax(repartition_train)
    rep_idx = [get_rep_idx(reps[k]) for k in range(n)]
    interesting_outputs = []
    for i,output in enumerate(rep_outputs[best_prompt_bench]):
        if len(output)>30:
            interesting_outputs.append((output,i))
    unique_interesting = []
    unique_idx = []
    for output,i in interesting_outputs:
        if output not in unique_interesting:
            unique_interesting.append(output)
            unique_idx.append(i)
    
    if print_graphs:
        print("Example repetition:")
        explain_rep(reps[best_prompt_bench][unique_idx[0]])
    
    #checking mems
    print("")
    print("Extraction for counterfactuals")
    mem_reps = [[] for k in range(n)]
    for k in range(n):
        for i,idx in enumerate(rep_idx[k]):
            if idx in real_mems_gen:
                mem_reps[k].append(reps[best_prompt_bench][i])
    repartition_mem = extr_results(mem_reps,bench_sizes,N_gen,"Extracted counterfactuals distribution",print_graphs)    

    
    #checking test
    print("")
    print("Extraction for test data")
    N_gen_test = len(test_gen["text"])
    min_length,bench = get_bench(bench_sizes,test_gen)
    if do_prompts:
        prompts = get_prompts(bench,test_gen)
        basic_reps,outputs = get_repetitions(prompts,test_gen,generator_gen,min_size=min_size,min_words=min_words,save=True,save_dir=f"{save_dir}/test_outputs",start_at=0,device=device)
    else:
        basic_reps = [[] for k in range(n)]
        outputs = [[] for k in range(n)]
        for i in range(n):
            with open(f"{save_dir}/test_outputs/outputs_{i}","r") as a:
                    lines = a.readlines()                
            outputs[i]=reconstruct_lines(lines,endline="[END]")
            with open(f"{save_dir}/test_outputs/repetitions_{i}","r") as a:
                lines = a.readlines()                
            basic_reps[i]=reconstruct_lines(lines,endline="[END]")
    
    reps = [get_unique_reps(basic_reps[k]) for k in range(n)]
    repartition_test = extr_results(reps,bench_sizes,N_gen,"Extracted in test dataset",print_graphs)
    
    plot_words_double(words_texts,words_extr_texts,"Initial texts")
    plot_words_double(words_prompt,words_extr_prompts,"Prompts")
    
    return repartition_train, repartition_test, repartition_mem, N_gen


def extr_grid(n_trials,dataset,real_mems,bench_sizes,seed=42,context_length=150,do_compute=True,do_training=True,max_new_tokens=20,min_size=25,min_words=7,save_dir="extr",print_graphs=True):
    
    if do_compute:
        print("Computing first extraction")
        t1 = perf_counter()
    repartition_train, repartition_test, repartition_mem, N_gen = extr_pipeline(dataset,real_mems,context_length=context_length,seed=seed,do_training=do_training,max_new_tokens=max_new_tokens,do_prompts=do_compute,bench_sizes=bench_sizes,min_size=min_size,min_words=min_words,save_dir=save_dir,print_graphs=True)
    
    if do_compute:
        utils.get_delay(t1)
        print("Computing grid extractions")
        t1 = perf_counter()
        repartition_trains = np.array(repartition_train)
        repartition_tests = np.array(repartition_test)
        repartition_mems = np.array(repartition_mem)

        for i in range(n_trials-1):
            print("-----------")
            print(f"Computing trial {i+1}")
            repartition_train, repartition_test, repartition_mem, _ = extr_pipeline(dataset,real_mems,context_length=context_length,seed=seed,do_training=do_training,max_new_tokens=max_new_tokens,do_prompts=True,bench_sizes=bench_sizes,min_size=min_size,min_words=min_words,save_dir=save_dir,print_graphs=False)
            repartition_trains += np.array(repartition_train)
            repartition_tests += np.array(repartition_test)
            repartition_mems += np.array(repartition_mem)
        utils.get_delay(t1)
        
        np.savetxt(f"{save_dir}/total_repartition_train.txt",repartition_trains)
        np.savetxt(f"{save_dir}/total_repartition_test.txt",repartition_tests)
        np.savetxt(f"{save_dir}/total_repartition_mem.txt",repartition_mems)
    
    else:
        repartition_trains = np.loadtxt(f"{save_dir}/total_repartition_train.txt")
        repartition_tests = np.loadtxt(f"{save_dir}/total_repartition_test.txt")
        repartition_mems = np.loadtxt(f"{save_dir}/total_repartition_mem.txt")
        
    print("")
    print("Train")
    plot_extr_results(repartition_trains/n_trials,"Extracted in training dataset",bench_sizes,N_gen)
    print("")
    print("Test")
    plot_extr_results(repartition_tests/n_trials,"Extracted in test dataset",bench_sizes,N_gen)
    print("")
    print("Counterfactuals")
    plot_extr_results(repartition_mems/n_trials,"Extracted in counterfactuals",bench_sizes,N_gen)
    
    return repartition_trains, repartition_tests, repartition_mems