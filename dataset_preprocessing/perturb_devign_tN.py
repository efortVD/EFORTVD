import random
import pandas as pd
import perturbation_builder
import os
import numpy as np
import transformations
from perturbation_builder import estimate_token_count
## Load pre-trained tokenizersimport transformations




# ============================================================
# CONFIGURAZIONE
# ============================================================
SEED = 42

# Set random seeds for reproducibility
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

model_size = 1024
devign_path = f'./perturbed_devign_{model_size}'




test_set_type_2 = pd.read_json(f'{devign_path}/type2/test.json')['func'].tolist()
test_set_t2_1 = pd.read_json(f'{devign_path}/transformations/T2_1/test.json')['func'].tolist()
test_set_t2_2 = pd.read_json(f'{devign_path}/transformations/T2_2/test.json')['func'].tolist()


test_set_type_2 = pd.read_json(f'{devign_path}/type2/test.json')['func'].tolist()
test_set_t2_1 = pd.read_json(f'{devign_path}/transformations/T2_1/test.json')['func'].tolist()
test_set_t2_2 = pd.read_json(f'{devign_path}/transformations/T2_2/test.json')['func'].tolist()


number_of_trasformations = 13

train_set = pd.read_json(f'{devign_path}/clean/train.json')
test_set = pd.read_json(f'{devign_path}/clean/test.json')


trasformations_name = ['T1_1', 'T1_2', 'T2_1', 'T2_2', 'T3_1', 'T3_2', 'T3_3', 'T3_4']
t1_1 = perturbation_builder.add_whitespace
t1_2 = perturbation_builder.add_comments

t2_1 = perturbation_builder.rename_variables
t2_2 = transformations.tf_3

t3_1 = transformations.tf_2
t3_2 = transformations.tf_8
t3_3 = perturbation_builder.add_dead_code
t3_4 = perturbation_builder.add_logging

trasformations = [t1_1, t1_2, t2_1, t2_2, t3_1, t3_2, t3_3, t3_4]

trasformations = [(name, func) for name, func in zip(trasformations_name, trasformations)]


os.makedirs(f'{devign_path}/transformations', exist_ok=True)


for name, tf in trasformations:
    print(f"Applying perturbation {name}...")
    os.makedirs(f'{devign_path}/transformations/{name}', exist_ok=True)

    train_set_transformation = train_set.copy()
    test_set_transformation = test_set.copy()
    if name == 'T3_1':
        train_set_transformation['func'] = train_set_transformation.apply(lambda row: tf(row['func']), axis=1)
        test_set_transformation['func'] = test_set_transformation.apply(lambda row: tf(row['func']), axis=1)
    elif name == 'T3_2':
        train_set_transformation['func'] = train_set_transformation.apply(lambda row: tf(row['func'], available=model_size - estimate_token_count(row['func']), buffer_tokens=5), axis=1)
        test_set_transformation['func'] = test_set_transformation.apply(lambda row: tf(row['func'], available=model_size - estimate_token_count(row['func']), buffer_tokens=5), axis=1)
    elif name == 'T3_3':
        train_set_transformation['func'] = train_set_transformation.apply(lambda row: tf(row['func'], available=model_size - estimate_token_count(row['func']), buffer_tokens=5), axis=1)
        test_set_transformation['func'] = test_set_transformation.apply(lambda row: tf(row['func'], available=model_size - estimate_token_count(row['func']), buffer_tokens=5), axis=1)
    elif name == 'T3_4':
        train_set_transformation['func'] = train_set_transformation.apply(lambda row: tf(row['func'], label=row['target'], available=model_size - estimate_token_count(row['func']), buffer_tokens=5), axis=1)
        test_set_transformation['func'] = test_set_transformation.apply(lambda row: tf(row['func'], label=row['target'], available=model_size - estimate_token_count(row['func']), buffer_tokens=5), axis=1)
    elif name == 'T1_1':
        train_set_transformation['func'] = train_set_transformation.apply(lambda row: tf(row['func'], available=model_size - estimate_token_count(row['func']), buffer_tokens=5), axis=1)
        test_set_transformation['func'] = test_set_transformation.apply(lambda row: tf(row['func'], available=model_size - estimate_token_count(row['func']), buffer_tokens=5), axis=1)
    elif name == 'T1_2':
        train_set_transformation['func'] = train_set_transformation.apply(lambda row: tf(row['func'], label=row['target'], available=model_size - estimate_token_count(row['func']), buffer_tokens=5), axis=1)
        test_set_transformation['func'] = test_set_transformation.apply(lambda row: tf(row['func'], label=row['target'], available=model_size - estimate_token_count(row['func']), buffer_tokens=5), axis=1)
    elif name == 'T2_1' or name == 'T2_2':
        train_set_transformation['func'] = train_set_transformation.apply(lambda row: tf(row['func']), axis=1)
        test_set_transformation['func'] = test_set_transformation.apply(lambda row: tf(row['func']), axis=1)
    

    train_set_transformation.to_json(f'{devign_path}/transformations/{name}/train.json', orient='records', lines=False)
    test_set_transformation.to_json(f'{devign_path}/transformations/{name}/test.json', orient='records', lines=False)

    print("Perturbation completed and saved.\n")

