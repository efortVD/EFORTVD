import pandas as pd
import perturbation_builder
import os
model_size = 2048
big_vul_path = f'./perturbed_big_vul_{model_size}'



test_csv = pd.read_csv(f'{big_vul_path}/clean/test.csv')






test_set_type4_clean = pd.read_json(f'{big_vul_path}/clean_subset_type4/test.json')
train_set_type4_clean = pd.read_json(f'{big_vul_path}/clean_subset_type4/train_subset.json')

test_set_type4 = pd.read_json(f'{big_vul_path}/type4/test.json')
train_set_type4 = pd.read_json(f'{big_vul_path}/type4/train_subset.json')

test_set_type4_func_before = test_set_type4['func_before'].tolist()
train_set_type4_func_before = train_set_type4['func_before'].tolist()

test_set_type4_func_before_clean = test_set_type4_clean['func_before'].tolist()
train_set_type4_func_before_clean = train_set_type4_clean['func_before'].tolist()

os.makedirs(f'{big_vul_path}', exist_ok=True)

train_set = pd.read_json(f'{big_vul_path}/clean/train_subset.json')
test_set = pd.read_csv(f'{big_vul_path}/clean/test.csv')

# make dir type1 typ2 type3 under perturb_big_vul to save the perturbed data
os.makedirs(f'{big_vul_path}/type1', exist_ok=True)
os.makedirs(f'{big_vul_path}/type2', exist_ok=True)
os.makedirs(f'{big_vul_path}/type3', exist_ok=True)

train_set_type1 = train_set.copy()
train_set_type2 = train_set.copy()
train_set_type3 = train_set.copy()

test_set_type1 = test_set.copy()
test_set_type2 = test_set.copy()
test_set_type3 = test_set.copy()

# apply type 1 perturbation on all samples
#for i in range(len(m4)):
#    m4.iloc[i]['func'] = generate_type3_clone(generate_type2_clone(generate_type1_clone(m4.iloc[i]['func'])))
train_set_type1['func_before'] = train_set_type1.apply(lambda row: perturbation_builder.generate_type1_clone(row['func_before'], label=row['target'], max_tokens=model_size), axis=1)
#train_set_type2 = train_set_type1.copy()
train_set_type2['func_before'] = train_set_type2.apply(lambda row: perturbation_builder.generate_type2_clone(row['func_before'], label=row['target'], just_this_type=False, max_tokens=model_size), axis=1)
#train_set_type3 = train_set_type2.copy()
train_set_type3['func_before'] = train_set_type3.apply(lambda row: perturbation_builder.generate_type3_clone(row['func_before'], label=row['target'], just_this_type=False, max_tokens=model_size), axis=1)
test_set_type1['func_before'] = test_set_type1.apply(lambda row: perturbation_builder.generate_type1_clone(row['func_before'], label=row['target'], max_tokens=model_size), axis=1)
#test_set_type2 = test_set_type1.copy()
test_set_type2['func_before'] = test_set_type2.apply(lambda row: perturbation_builder.generate_type2_clone(row['func_before'], label=row['target'], just_this_type=False, max_tokens=model_size), axis=1)
#test_set_type3 = test_set_type2.copy()
test_set_type3['func_before'] = test_set_type3.apply(lambda row: perturbation_builder.generate_type3_clone(row['func_before'], label=row['target'], just_this_type=False, max_tokens=model_size), axis=1)

train_set_type1.to_json(f'{big_vul_path}/type1/train_subset.json', orient='records', lines=False)
train_set_type2.to_json(f'{big_vul_path}/type2/train_subset.json', orient='records', lines=False)
train_set_type3.to_json(f'{big_vul_path}/type3/train_subset.json', orient='records', lines=False)

test_set_type1.to_json(f'{big_vul_path}/type1/test.json', orient='records', lines=False)
test_set_type2.to_json(f'{big_vul_path}/type2/test.json', orient='records', lines=False)
test_set_type3.to_json(f'{big_vul_path}/type3/test.json', orient='records', lines=False)

print("Perturbation completed and saved.")



os.makedirs(f'{big_vul_path}/type4', exist_ok=True)

# read all files.ccp (func and func_refactored) from type4/functions_big_vul and create test.json and train.json
import glob
funcs = {}
funcs_refactored = {}
#order the files in numerical order (func_0 func_1 func_2 ... func_10 func_11 ... )
file_ids = []
files_ids_refactored = []
filepaths = glob.glob('../type4/functions_big_vul/train/*.cpp')
# Sort by the numeric ID extracted from filename
for filepath in sorted(filepaths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])):
    # get id func_0.cpp -> 0
    
    with open(filepath, 'r') as f:
        content = f.read()
        if 'refactored' in filepath:
            file_id = int(os.path.basename(filepath).split('_')[1].split('.')[0])
            files_ids_refactored.append(file_id)
            funcs_refactored[file_id] = content
        else:
            file_id = int(os.path.basename(filepath).split('_')[1].split('.')[0])
            file_ids.append(file_id)
            funcs[file_id] = content

assert file_ids == files_ids_refactored, "Mismatch between original and refactored file IDs"

#train_subset = pd.read_json('../data/perturbed_big_vul_2048/clean/train_subset.json')

train_set = pd.read_csv('../StagedVulBERT/resource/dataset/train.csv', nrows=10000)



targets = train_set['target'].tolist()
train_set_filtered = train_set.iloc[file_ids].copy()


count_0_labels = sum(train_set_filtered['target'] == 0)
count_1_labels = sum(train_set_filtered['target'] == 1)

# i want 1000 (418 for label 1 and 582 for label 0) samples in type4 train set
desired_count_1 = count_1_labels
desired_count_0 = 1000 - desired_count_1
indices_to_keep = []
count_0 = 0
count_1 = 0
for idx, row in train_set.iterrows():
    if idx in file_ids:
        if row['target'] == 1 and count_1 < desired_count_1:
            indices_to_keep.append(idx)
            count_1 += 1
        elif row['target'] == 0 and count_0 < desired_count_0:
            indices_to_keep.append(idx)
            count_0 += 1

        
        if count_0 >= desired_count_0 and count_1 >= desired_count_1:
            break

with open(f'{big_vul_path}/type4/train_indices.txt', 'w') as f:
    for idx in indices_to_keep:
        f.write(f"{idx}\n")

funcs_refactored_filtered = [funcs_refactored[i] for i in funcs_refactored.keys() if i in indices_to_keep]

funcs_before_train_set = train_set['func_before'].tolist()

funcs_before_train_set_filtered = train_set_filtered['func_before'].tolist()    


train_set_type4 = train_set.loc[indices_to_keep].reset_index(drop=True).copy()

funcs_before_type4_train = train_set_type4['func_before'].tolist()
print(len(train_set_type4))
for i in range(len(train_set_type4)):
    train_set_type4.loc[i, 'func_before'] = funcs_refactored_filtered[i]

funcs = {}
funcs_refactored = {}
#order the files in numerical order (func_0 func_1 func_2 ... func_10 func_11 ... )
file_ids = []
files_ids_refactored = []
filepaths = glob.glob('../type4/functions_big_vul/test/*.cpp')
# Sort by the numeric ID extracted from filename
for filepath in sorted(filepaths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])):
    # get id func_0.cpp -> 0
    
    with open(filepath, 'r') as f:
        content = f.read()
        if 'refactored' in filepath:
            file_id = int(os.path.basename(filepath).split('_')[1].split('.')[0])
            files_ids_refactored.append(file_id)
            funcs_refactored[file_id] = content
        else:
            file_id = int(os.path.basename(filepath).split('_')[1].split('.')[0])
            file_ids.append(file_id)
            funcs[file_id] = content

assert file_ids == files_ids_refactored, "Mismatch between original and refactored file IDs"

test_set = pd.read_csv(f'../data/perturbed_big_vul_{model_size}/clean/test.csv', nrows=10000)
#test_set = pd.read_csv('../StagedVulBERT/resource/dataset/test.csv', nrows=10000)


test_set_filtered = test_set.iloc[file_ids].copy()

count_0_labels = sum(test_set_filtered['target'] == 0)
count_1_labels = sum(test_set_filtered['target'] == 1)

# i want 1000 (418 for label 1 and 582 for label 0) samples in type4 train set
desired_count_1 = count_1_labels
desired_count_0 = 1000 - desired_count_1
indices_to_keep = []
count_0 = 0
count_1 = 0
for idx, row in test_set.iterrows():
    if idx in file_ids:
        if row['target'] == 1 and count_1 < desired_count_1:
            indices_to_keep.append(idx)
            count_1 += 1
        elif row['target'] == 0 and count_0 < desired_count_0:
            indices_to_keep.append(idx)
            count_0 += 1

        
        if count_0 >= desired_count_0 and count_1 >= desired_count_1:
            break

with open(f'{big_vul_path}/type4/test_indices.txt', 'w') as f:
    for idx in indices_to_keep:
        f.write(f"{idx}\n")

funcs_refactored_filtered = [funcs_refactored[i] for i in funcs_refactored.keys() if i in indices_to_keep]


test_set_type4 = test_set.loc[indices_to_keep].reset_index(drop=True).copy()
print(len(test_set_type4))
for i in range(len(test_set_type4)):
    test_set_type4.loc[i, 'func_before'] = funcs_refactored_filtered[i]



train_set_type4['func_before'] = train_set_type4.apply(lambda row: perturbation_builder.generate_type3_clone(row['func_before'], label=row['target'], just_this_type=False, max_tokens=model_size), axis=1)
test_set_type4['func_before'] = test_set_type4.apply(lambda row: perturbation_builder.generate_type3_clone(row['func_before'], label=row['target'], just_this_type=False, max_tokens=model_size), axis=1)

train_set_type4.to_json(f'{big_vul_path}/type4/train_subset.json', orient='records', lines=False) 
test_set_type4.to_json(f'{big_vul_path}/type4/test.json', orient='records', lines=False)

print("Original train set size:", len(train_set_type4))