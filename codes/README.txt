Main notebooks : 
"hoc.ipynb" : for BLUE dataset
"data_hcl.ipynb" : for models A' & B' (HCL & Antoine Richard) 
"patho.ipynb" : for pathology dataset (similar to BLUE, in French, from HCL)

Main python files :
"utils.py" : main utilities
"mem.py" : for counterfactual memorization
"extr.py" : for data extraction
"mia.py" : for membership inference
"dp.py" : for differential privacy
"fed.py" : for federated learning

Requires the following libraries to be installed (add ! in front of pip to execute in a notebook cell)

pip install torch
pip install transformers
pip install datasets
pip install graphviz
pip install evaluate
pip install opacus
pip install seaborn
pip install scikit-learn
pip install ipywidgets
pip install xgboost

#if running at hcl, use following command:
#!conda run -n <id> pip install <package> --proxy="http://<id>:<mdp>@proxy-ng.chu-lyon.fr:8080


possibles bugs :

- tqdm : si pas les bonnes versions de ipywidgets ou certains environnements jupyter différents, ne pas hésiter à enlever dans ce cas
permet de suivre progression des calculs, ajouter par exmeple :  for i in tqdm(range(n))
