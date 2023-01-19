import pandas as pd
import numpy as np
import tensorflow as tf
import deepchem as dc
import matplotlib.pyplot as plt
import time
import os
import shutil
import logging


logging.basicConfig(level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(123)
tf.compat.v1.set_random_seed(123)


def load_data(featurizer='ECFP', split='random', reload=True, delete_old_dataset=False, data_save_dir='./', data_dir=None, input_tasks=None):
    # Delete the previously saved featurized datasets to extract them again
    if not reload:
        if delete_old_dataset:
            for t in ['train', 'test', 'valid']:
                if os.path.isdir(data_save_dir + t + '_dir/'):
                    shutil.rmtree(data_save_dir + t + '_dir/')
    os.makedirs(data_save_dir, exist_ok=True)

    # assign data and tasks
    dataset_file = data_dir
    tasks = input_tasks
    valid_indices, test_indices = None, None
    if split == 'specified':
        dummy_df = pd.read_csv(data_dir, low_memory=True)
        valid_indices = dummy_df.index[dummy_df['split'] == 'validation'].tolist()
        test_indices = dummy_df.index[dummy_df['split'] == 'test'].tolist()
    print("About to load the dataset.")

    # create featurizer, loader, transformers, and splitter
    if featurizer == 'ECFP':
        featurizer = dc.feat.CircularFingerprint(size=1024, chiral=True)
    elif featurizer == 'GraphConv':
        featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    loader = dc.data.CSVLoader(tasks=tasks, feature_field="smiles", featurizer=featurizer)
    splitters = {
        'index': dc.splits.IndexSplitter(),
        'random': dc.splits.RandomSplitter(),
        'scaffold': dc.splits.ScaffoldSplitter(),
        'specified': dc.splits.SpecifiedSplitter(valid_indices=valid_indices, test_indices=test_indices)
    }
    splitter = splitters[split]

    # check if built dataset exists on disk
    found_flag = 0
    if reload:
        loaded, dataset, transformers = dc.utils.data_utils.load_dataset_from_disk(data_save_dir)
        if loaded:
            train_dataset, valid_dataset, test_dataset = dataset
            found_flag = 1
    # if the built dataset does not exist, create it
    if not found_flag:
        if not os.path.exists(dataset_file):
            print("Dataset not found")
        print("About to featurize the dataset.")
        dataset = loader.create_dataset([dataset_file], shard_size=8192)
        # Initialize transformers
        print("About to transform data")
        transformers = [dc.trans.BalancingTransformer(dataset=dataset)]
        for transformer in transformers:
            dataset = transformer.transform(dataset)
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)
        dc.utils.data_utils.save_dataset_to_disk(data_save_dir, train=train_dataset, valid=valid_dataset, test=test_dataset, transformers=transformers)
    return tasks, (train_dataset, valid_dataset, test_dataset), transformers, loader

# run_type = 'single'
# run_type = 'multicancer'
# run_type = 'multi'
run_type= 'zeroshot'
# run_type= 'zeroshot_roc'
# run_type= 'random'

# metric_valid = 'roc'
metric_valid = 'ap'
data_dir = 'data/pubchem_pcba_merged_splits_countered.csv'
main_df = pd.read_csv(data_dir, low_memory=False)
input_columns = list(main_df.columns)
target_task = 'PCBA-2289'
if run_type == 'single':
    input_tasks = [target_task]
elif run_type == 'multicancer':
    input_tasks = ['PCBA-' + str(i) for i in [
        1454,2147,2517,2549,2662,463254,485290,493208,504332,504467,504706,588453,588456,588579,588590,588591,588795,
        602179,624246,624296,624297,686970,686978,686979,720504,875,902,903,904,914,915,924,995,624202,651725,504444,
        602244,588664,2289,623901,504466,1452,881,624170,540317,504891,504327,588489]]
elif run_type == 'multi':
    input_tasks = list(np.array(input_columns)[[True if 'PCBA' in c else False for c in input_columns]])
elif run_type == 'zeroshot':
    input_tasks = ['PCBA-1458', 'PCBA-485297', 'PCBA-485313', 'PCBA-588342',
                   'PCBA-504466', 'PCBA-2289', 'PCBA-624202']
elif run_type == 'zeroshot_roc':
    input_tasks = ['PCBA-1379', 'PCBA-1458', 'PCBA-1721', 'PCBA-411', 'PCBA-485297',
                   'PCBA-485313', 'PCBA-504706', 'PCBA-588342', 'PCBA-504466',
                   'PCBA-2289', 'PCBA-493014', 'PCBA-624202']
elif run_type == 'random':
    input_tasks = ['PCBA-875', 'PCBA-720504', 'PCBA-881', 'PCBA-588579', 'PCBA-2517', 'PCBA-623901',
                   'PCBA-588795', 'PCBA-904', 'PCBA-2289']

target_index = input_tasks.index(target_task)
print(input_tasks)
print(target_index)
# Change reload dataset to False for the first time dataset is featurized, then true afterwards
reload = True
# If you are featurizing the same dataset again, the older dataset needs to be deleted.
# Would only work if reload = False
delete_old_dataset = False
split = 'specified'
featurizer = 'GraphConv'
timestr = time.strftime("%m%d-%H%M")
model_dir = 'built_models/mirna_cleaned/'+featurizer+'/' + timestr + '/'
if os.path.isdir(model_dir):
    timestr = timestr.split('-')[0] + '-' + timestr.split('-')[1][:2] + str(int(timestr.split('-')[1][2:])+60)
os.makedirs(model_dir, exist_ok=True)
data_save_dir = 'built_datasets/mirna_cleaned/'+featurizer+'/'+split+'/' + run_type + '/'

# Split dataset by scaffold into 80 10 10 percent splits
all_tasks, (train_dataset, valid_dataset, test_dataset), transformers, loader = load_data(data_save_dir=data_save_dir,featurizer=featurizer, split=split, reload=reload, delete_old_dataset=delete_old_dataset, data_dir=data_dir, input_tasks=input_tasks)
training_data_len = len(train_dataset.y)

print('Number of positive samples in test dataset', int(np.sum(test_dataset.y[:,target_index])))
print('Number of positive samples in valid dataset', int(np.sum(valid_dataset.y[:,target_index])))
print('Number of positive samples in train dataset', int(np.sum(train_dataset.y[:,target_index])))

if metric_valid == 'roc':
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode="classification", n_tasks=len(input_tasks))
elif metric_valid == 'ap':
    metric = dc.metrics.Metric(dc.metrics.prc_auc_score, mode="classification", n_tasks=len(input_tasks))

params_dict = {
    'n_tasks': [len(input_tasks)],
    'graph_conv_layers': [[64], [64,64,64], [256], [512,512,512]],
    'dropouts': [0, 0.1],
    'learning_rate': [0.001, 0.0001],
    'batch_size': [32, 128],
    'number_atom_features': [78],
    'dense_layer_size': [256, 1024]
}

optimizer = dc.hyper.GridHyperparamOpt(dc.models.GraphConvModel)
best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict, train_dataset, valid_dataset, metric, transformers, nb_epoch=10, use_max=True, logdir=model_dir)

results = []
labels = []
for r in all_results:
    labels.append(r)
    results.append(all_results[r])

cleaned_results = pd.DataFrame(data={'parameter': np.array(results), 'ap': labels})
cleaned_results.to_csv('results/hypeopt_'+run_type+'_'+metric_valid+'.csv', header=True, index=False)

fig = plt.figure(figsize=(5,4), dpi=300)
plt.plot(range(1,len(results)+1), results, 'o-')
plt.xlabel('Parameter Set')
if metric_valid == 'roc':
    plt.ylabel('Validation ROCAUC')
elif metric_valid == 'ap':
    plt.ylabel('Validation Average Precision Score')
plt.tight_layout()
plt.savefig('results/hyper_opt_'+run_type+'_'+metric_valid+'.png', format='png',dpi=300)
plt.show()