import json
import random
import pandas as pd
import perturbation_builder
import os
import numpy as np


model_size = '1024'
devign_path = f'./perturbed_devign_{model_size}'

clean_test = pd.read_json(f'{devign_path}/clean/test.json')['func'].tolist()
clean_train = pd.read_json(f'{devign_path}/clean/train.json')['func'].tolist()

test_set = pd.read_json(f'{devign_path}/type3/test.json')['func'].tolist()
train_set = pd.read_json(f'{devign_path}/type3/train.json')['func'].tolist()


mydata = pd.read_json(f'{devign_path}/clean/mydata.json')
train_idx_devign=set()
test_idx_devign=set()
with open(f'{devign_path}/clean/train_indices.txt') as f:
    for line in f:
        line=line.strip()
        train_idx_devign.add(int(line))
with open(f'{devign_path}/clean/test_indices.txt') as f:
    for line in f:
        line=line.strip()
        test_idx_devign.add(int(line))
train_set_or = mydata.iloc[list(train_idx_devign)]
test_set_or = mydata.iloc[list(test_idx_devign)]

train_idx_devign = list(train_idx_devign)
test_idx_devign = list(test_idx_devign)

indices_train = train_set_or.index.tolist()
indices_test = test_set_or.index.tolist()

# save indices
with open(f'{devign_path}/clean/train_indices.txt', 'w') as f:
    for idx in indices_train:
        f.write(f"{idx}\n")
with open(f'{devign_path}/clean/test_indices.txt', 'w') as f:
    for idx in indices_test:
        f.write(f"{idx}\n")

train_set = pd.read_json(f'{devign_path}/clean/train.json')
test_set = pd.read_json(f'{devign_path}/clean/test.json')

train_set_old = pd.read_json(f'{devign_path}/clean/train.json')
test_set_old = pd.read_json(f'{devign_path}/clean/test.json')

# read train and test indices for type 4
with open(f'{devign_path}/type4/test_indices.txt') as f:
    test_indices_type4 = [int(line.strip()) for line in f.readlines()]

with open(f'{devign_path}/type4/train_indices.txt') as f:
    train_indices_type4 = [int(line.strip()) for line in f.readlines()]

train_set_clean_type4 = pd.read_json(f'{devign_path}/clean_type4/train.json')
test_set_clean_type4 = pd.read_json(f'{devign_path}/clean_type4/test.json')
labels_test_set_or = {idx: label for idx, label in zip(test_idx_devign, test_set_or['target'].tolist())}
labels_test_set_or2 = [labels_test_set_or[idx] for idx in test_indices_type4]
labels_test_set = test_set['target'].tolist()
clean_labels_or = {idx: label for idx, label in zip(test_idx_devign, labels_test_set)}

train_set_clean_type4_or = mydata.iloc[train_indices_type4]
test_set_clean_type4_or = mydata.iloc[test_indices_type4]

test_set_clean_type4_or_labels = test_set_clean_type4_or['target'].tolist()
test_set_clean_type4_label = test_set_clean_type4['target'].tolist()

import pickle
clean_pred = pickle.load(open(f'{devign_path}/results_{model_size}/clean_test_pred.pkl', 'rb'))
clean_labels = pickle.load(open(f'{devign_path}/results_{model_size}/clean_test_labels.pkl', 'rb'))
clean_pred = {idx: pred for idx, pred in zip(test_idx_devign, clean_pred)}
clean_labels = {idx: label for idx, label in zip(test_idx_devign, clean_labels)}
# Load perturbed data
perturbed_pred = pickle.load(open(f'{devign_path}/results_{model_size}/type3_test_perturbed_pred.pkl', 'rb'))
perturbed_labels = pickle.load(open(f'{devign_path}/results_{model_size}/type3_test_perturbed_labels.pkl', 'rb'))

perturbed_pred = {idx: pred for idx, pred in zip(test_idx_devign, perturbed_pred)}
perturbed_labels = {idx: label for idx, label in zip(test_idx_devign, perturbed_labels)}


clean_pred_list = [clean_pred[i] for i in test_indices_type4]
clean_labels_list = [clean_labels[i] for i in test_indices_type4]

perturbed_pred_type1 = [perturbed_pred[i] for i in test_indices_type4]
perturbed_labels_type1 = [perturbed_labels[i] for i in test_indices_type4]
# ============================================================
# CONFIGURAZIONE
# ============================================================
SEED = 42

# Set random seeds for reproducibility
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)



os.makedirs(f'{devign_path}/type1', exist_ok=True)
os.makedirs(f'{devign_path}/type2', exist_ok=True)
os.makedirs(f'{devign_path}/type3', exist_ok=True)
os.makedirs(f'{devign_path}/type4', exist_ok=True)

train_set = pd.read_json(f'{devign_path}/clean/train.json')
test_set = pd.read_json(f'{devign_path}/clean/test.json')




train_set_type1 = train_set.copy()
train_set_type2 = train_set.copy()
train_set_type3 = train_set.copy()

test_set_type1 = test_set.copy()
test_set_type2 = test_set.copy()
test_set_type3 = test_set.copy()

# apply type 1 perturbation on all samples
#for i in range(len(m4)):
#    m4.iloc[i]['func'] = generate_type3_clone(generate_type2_clone(generate_type1_clone(m4.iloc[i]['func'])))
train_set_type1['func'] = train_set_type1.apply(lambda row: perturbation_builder.generate_type1_clone(row['func'], label=row['target'], max_tokens=model_size), axis=1)
#train_set_type2 = train_set_type1.copy()
train_set_type2['func'] = train_set_type2.apply(lambda row: perturbation_builder.generate_type2_clone(row['func'], label=row['target'], just_this_type=False, max_tokens=model_size), axis=1)
#train_set_type3 = train_set_type2.copy()
train_set_type3['func'] = train_set_type3.apply(lambda row: perturbation_builder.generate_type3_clone(row['func'], label=row['target'], just_this_type=False, max_tokens=model_size), axis=1)
test_set_type1['func'] = test_set_type1.apply(lambda row: perturbation_builder.generate_type1_clone(row['func'], label=row['target'], max_tokens=model_size), axis=1)
#test_set_type2 = test_set_type1.copy()
test_set_type2['func'] = test_set_type2.apply(lambda row: perturbation_builder.generate_type2_clone(row['func'], label=row['target'], just_this_type=False, max_tokens=model_size), axis=1)
#test_set_type3 = test_set_type2.copy()
test_set_type3['func'] = test_set_type3.apply(lambda row: perturbation_builder.generate_type3_clone(row['func'], label=row['target'], just_this_type=False, max_tokens=model_size), axis=1)

train_set_type1.to_json(f'{devign_path}/type1/train.json', orient='records', lines=False)
train_set_type2.to_json(f'{devign_path}/type2/train.json', orient='records', lines=False)
train_set_type3.to_json(f'{devign_path}/type3/train.json', orient='records', lines=False) 
test_set_type1.to_json(f'{devign_path}/type1/test.json', orient='records', lines=False)
test_set_type2.to_json(f'{devign_path}/type2/test.json', orient='records', lines=False)
test_set_type3.to_json(f'{devign_path}/type3/test.json', orient='records', lines=False)

print("Perturbation completed and saved.")


# read all files.ccp (func and func_refactored) from type4/functions_devign and create test.json and train.json
import glob
funcs = {}
funcs_refactored = {}
#order the files in numerical order (func_0 func_1 func_2 ... func_10 func_11 ... )
file_ids = []
filepaths = glob.glob('../type4/functions_devign/*.cpp')
# Sort by the numeric ID extracted from filename
for filepath in sorted(filepaths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])):
    # get id func_0.cpp -> 0
    
    with open(filepath, 'r') as f:
        content = f.read()
        if 'refactored' in filepath:
            file_id = int(os.path.basename(filepath).split('_')[1].split('.')[0])
            funcs_refactored[file_id] = content
        else:
            file_id = int(os.path.basename(filepath).split('_')[1].split('.')[0])
            file_ids.append(file_id)
            funcs[file_id] = content

funcs_keys = set(funcs.keys())
funcs_refactored_keys = set(funcs_refactored.keys())
print(f"Total funcs: {len(funcs_keys)}, refactored funcs: {len(funcs_refactored_keys)}")
missing_refactored = funcs_keys - funcs_refactored_keys
if missing_refactored:
    print(f"Missing refactored files for IDs: {missing_refactored}")

train_index=set()
test_index=set()
with open('./finetune/devign/train.txt') as f:
    for line in f:
        line=line.strip()
        train_index.add(int(line))
with open('./finetune/devign/test.txt') as f:
    for line in f:
        line=line.strip()
        test_index.add(int(line))

mydata = pd.read_json('./finetune/devign/Devign.json')
# just take the indexes in file_ids to keep the order

train_index_after = [t for t in train_index if t in file_ids]
test_index_after = [t for t in test_index if t in file_ids]

train_list = list(train_index_after)
test_list = list(test_index_after)
# save the indeces used for train and test set
with open(f'{devign_path}/type4/train_indices.txt', 'w') as f:
    for idx in list(train_index_after):
        f.write(f"{idx}\n")
with open(f'{devign_path}/type4/test_indices.txt', 'w') as f:
    for idx in list(test_index_after):
        f.write(f"{idx}\n")


train_set_type4 = mydata.iloc[train_list].copy()
test_set_type4 = mydata.iloc[test_list].copy()

count_0_train = train_set_type4['target'].tolist().count(0)
count_1_train = train_set_type4['target'].tolist().count(1)
count_0_test = test_set_type4['target'].tolist().count(0)
count_1_test = test_set_type4['target'].tolist().count(1)

# replace func with the content from funcs according to the index in file_ids
train_set_type4 = train_set_type4.reset_index(drop=True)
for i in range(len(train_set_type4)):
    idx = train_list[i]
    file_idx = file_ids.index(idx)
    train_set_type4.loc[i, 'func'] = funcs_refactored[idx]

test_set_type4 = test_set_type4.reset_index(drop=True)
for i in range(len(test_set_type4)):
    idx = test_list[i]
    file_idx = file_ids.index(idx)
    test_set_type4.loc[i, 'func'] = funcs_refactored[idx]



# Apply type3 transformation with label
train_set_type4['func'] = train_set_type4.apply(lambda row: perturbation_builder.generate_type3_clone(row['func'], label=row['target'], just_this_type=False, max_tokens=model_size), axis=1)
test_set_type4['func'] = test_set_type4.apply(lambda row: perturbation_builder.generate_type3_clone(row['func'], label=row['target'], just_this_type=False, max_tokens=model_size), axis=1)

#test_set_type3 = pd.read_json('./perturbed_devign/type3/test.json')


train_set_type4.to_json(f'{devign_path}/type4/train.json', orient='records') 
test_set_type4.to_json(f'{devign_path}/type4/test.json', orient='records')


print("Original train set size:", len(train_set_type4))
