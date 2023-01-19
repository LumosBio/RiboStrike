import pandas as pd
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.compat.v1.set_random_seed(123)
import deepchem as dc
from rdkit.Chem import MolFromSmiles, MolToSmiles
import logging


logging.basicConfig(level=logging.INFO)

def canon_smile(smile):
    return MolToSmiles(MolFromSmiles(smile), isomericSmiles=True)

# Folder that contains the dataset and will contain the preprocessed datasets
data_dir = 'data/'

file_names = ['pcba']

# These are the source files, please keep the same
df = pd.read_csv(data_dir+file_names[0]+'.csv', low_memory=False)
print('Initial shape: ', df.shape)
print(df.columns)

# Delete empty or duplicate smiles
df = df.dropna(subset=['smiles'])
df = df.drop_duplicates(subset='smiles', keep='first')
df.reset_index(inplace=True)
print('Shape after deleting empty or duplicate smiles: ', df.shape)

# Find unique values in columns
df_filled = df.fillna(value=0)
phenotype_column = None
outcome_columns = []
for column in df.columns:
    if 'PCBA' in column:
        outcome_columns.append(column)
        print(column, np.unique(df_filled[column]))

# Make all smiles canon
smiles = df['smiles']
smiles_canon = [canon_smile(s) for s in smiles]
df['smiles'] = smiles_canon
print('smiles and unique smiles: ', len(smiles_canon), len(np.unique(smiles_canon)))

# Shuffle and save
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(data_dir+ file_names[0]+'_cleaned.csv', header=True, index=False)

# Find shared tasks between miR-21 and PCBA
task_numbers = []
for o in outcome_columns:
    task_numbers.append(o.split('-')[1])

target_files = [
'193771_AID_1285_datatable_inhibit Alzheimer\'s amyloid precursor protein (APP) translation Alzheimer.csv',
'279988_AID_2675_datatable_type I myotonic dystrophy (DM1).csv',
'309031_AID_2551_datatable_all.csv',
'321735_AID_624417_datatable_all.csv',
'330308_AID_504466_datatable_ATAD5_ genotoxicity.csv',
'331108_AID_485367_datatable_Inhibitors of T. brucei phosphofructokinase.csv',
'335997_AID_623901_datatable_miR122 activator anticancer.csv',
'336623_AID_2289_datatableall_df_Inhibitors of miR-21 anticancer.csv',
'337074_AID_493014_datatable_non-Hodgkin lymphoma anticancer.csv',
'337903_AID_485314_datatable_Inhibitors of DNA Polymerase Beta.csv',
'340928_AID_493160_datatable_inhibitors of hexokinase domain containing I (HKDC1) diabetes.csv',
'340929_AID_493091_datatable_identification of inhibitors of Scp-1.csv',
'343600_AID_651820_datatable_Inhibitors of Hepatitis C Virus.csv',
'351226_AID_588855_datatable_all.csv',
'359484_AID_588664_datatable_inhibitors of the interaction of the Ras and Rab interactor 1 protein (Rin1) anticancer.csv',
'359521_AID_588489_datatable_ 588489 RNA.csv',
'362388_AID_602244_datatable_CXCR6 Inhibitors in a B-arrestin anticancer.csv',
'364195_AID_504444_datatable_Nrf2 qHTS screen for inhibitors anticancer.csv',
'364407_AID_651725_datatable_Inhibitors of the Six1:Eya2 Interaction anticancer.csv',
'377550_AID_624202_datatable_Activators of BRCA1 Expression anticancer.csv']
target_numbers_1 = []
target_numbers_2 = []
for f in target_files:
    target_numbers_1.append(f.split('_')[0])
    target_numbers_2.append(f.split('_')[2])

target_dict = {}
for i in range(len(target_numbers_1)):
    target_dict[target_numbers_1[i]] = target_numbers_2[i]


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

print(intersection(target_numbers_1, task_numbers))
print(intersection(target_numbers_2, task_numbers))
intersect = intersection(target_numbers_2, task_numbers)

# Delete shared tasks from PCBA
counter = 0
for i in range(len(target_numbers_2)):
    print(counter, '-', target_numbers_2[i])
    counter += 1
for i in range(len(task_numbers)):
    if task_numbers[i] not in intersect:
        print(counter, '-', task_numbers[i])
        counter += 1
overlap_columns = []
for o in outcome_columns:
    for i in intersect:
        if i in o:
            overlap_columns.append(o)
for o in target_files:
    for i in intersect:
        if i in o:
            print(o)

df = df.drop(overlap_columns, axis=1)
print(df.shape)
df.to_csv(data_dir+ file_names[0]+'_cleaned_no_overlap.csv', header=True, index=False)

merged = pd.read_csv(data_dir+'pubchem/pubchem_merged.csv', low_memory=True)


# Find overlap between smiles of all datasets with the target dataset
target_smiles = np.array(merged['smiles'])
current_smiles = np.array(df['smiles'])
print(file_names[0], 'smiles overlap percentage: ', len(np.intersect1d(target_smiles, current_smiles))/len(current_smiles)*100)

# Merge all dataframes using unique smiles
merged = df.merge(merged, on=['smiles'], how='outer')
print(merged.head())
print(merged.shape)
print(merged.iloc[0])
smiles_canon = merged['smiles']
print('smiles and unique smiles: ', len(smiles_canon), len(np.unique(smiles_canon)))
print('number of rows with missing data: ', merged.shape[0] - merged.dropna().shape[0])
merged.to_csv(data_dir+'pubchem_pcba_merged.csv', header=True, index=False)

print(merged.columns)
input_data = data_dir+'pubchem_pcba_merged.csv'
merged = pd.read_csv(input_data, low_memory=False)
columns_dict = {}
for c in merged.columns:
    if c.split('_')[-1] in target_dict:
        columns_dict[c] = 'PCBA-' + target_dict[c.split('_')[-1]]
# columns_dict = {}
# Rename mis-labeled columns
columns_dict['PUBCHEM_ACTIVITY_OUTCOME_x'] = 'PCBA-1285'
columns_dict['PUBCHEM_ACTIVITY_OUTCOME_y'] = 'PCBA-2675'
merged = merged.rename(columns=columns_dict)
print(merged.columns)
print(merged.columns[-13])

# Count the non-missing data in each task
counter = 0
chosen_columns = []
data_counts = []
for c in merged.columns:
    if 'PCBA' in c:
        chosen_columns.append(c.split('-')[-1])
        data_counts.append(merged[c].dropna().shape[0])
        print(counter, '-', c, merged[c].dropna().shape[0])
        counter += 1
arash_df = pd.DataFrame(data={'AID Number': np.array(chosen_columns), 'Data Count': np.array(data_counts)})
arash_df.to_csv(data_dir+'data_count.csv', header=True, index=False)

merged.to_csv(data_dir+'pubchem_pcba_merged.csv', header=True, index=False)

# Split the data
input_data = data_dir+'pubchem_pcba_merged.csv'
input_columns = list(pd.read_csv(input_data, low_memory=True).columns)
input_tasks = []
for c in input_columns:
    if 'PCBA' in c:
        input_tasks.append(c)
print(input_tasks)
print(len(input_tasks))

split = 'scaffold'
featurizer = 'GraphConv'
reload_dataset = False
delete_old_dataset = True
import os
import shutil
data_save_dir = 'built_datasets/mirna_cleaned/'+featurizer+'/'+split+'/'
# Delete the previously saved featurized datasets to extract them again
if not reload_dataset:
    if delete_old_dataset:
        for t in ['train', 'test', 'valid']:
            if os.path.isdir(data_save_dir+t+'_dir/'):
                shutil.rmtree(data_save_dir+t+'_dir/')
os.makedirs(data_save_dir, exist_ok=True)

def load_mir21(featurizer='ECFP', split='random', reload=reload_dataset, data_save_dir=data_save_dir, input_data=input_data, input_tasks=input_tasks):
    # assign data and tasks
    dataset_file = input_data
    mir21_tasks = input_tasks
    valid_indices, test_indices = None, None
    if split == 'specified':
        dummy_df = pd.read_csv(input_data, low_memory=True)
        valid_indices = dummy_df.index[dummy_df['split'] == 'validation'].tolist()
        test_indices = dummy_df.index[dummy_df['split'] == 'test'].tolist()
    print("About to load the dataset.")

    # create featurizer, loader, transformers, and splitter
    if featurizer == 'ECFP':
        featurizer = dc.feat.CircularFingerprint(size=1024, chiral=True)
    elif featurizer == 'GraphConv':
        featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    loader = dc.data.CSVLoader(tasks=mir21_tasks, feature_field="smiles", featurizer=featurizer)
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
        # dataset = loader.create_dataset().featurize(dataset_file, shard_size=8192)
        dataset = loader.create_dataset([dataset_file], shard_size=8192)
        # Initialize transformers
        print("About to transform data")
        transformers = [dc.trans.BalancingTransformer(dataset=dataset)]
        for transformer in transformers:
            dataset = transformer.transform(dataset)
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)
        dc.utils.data_utils.save_dataset_to_disk(data_save_dir, train=train_dataset, valid=valid_dataset, test=test_dataset, transformers=transformers)
        # print("Columns of dataset: %s" % str(dataset.columns.values))
        # print("Number of examples in dataset: %s" % str(dataset.shape[0]))
        # print(mir21_tasks,'is our task(s)')

    return mir21_tasks, (train_dataset, valid_dataset, test_dataset), transformers, loader


# Split dataset by scaffold into 80 10 10 percent splits
mir21_tasks, (train_dataset, valid_dataset, test_dataset), transformers, loader = load_mir21(featurizer=featurizer, split=split, input_data=input_data, input_tasks=input_tasks)

# Extract the smiles for each split
train_smiles = np.array(train_dataset.ids)
valid_smiles = np.array(valid_dataset.ids)
test_smiles = np.array(test_dataset.ids)

# Save the smiles back into the CSV file
all_smiles = np.array(list(train_smiles) + list(valid_smiles) + list(test_smiles))
labels_smiles = np.array(['train'] * len(train_smiles) + ['validation'] * len(valid_smiles) + ['test'] * len(test_smiles))
smiles_df = pd.DataFrame(data={'smiles': all_smiles, 'split':labels_smiles})
print(smiles_df.head())
merged = pd.read_csv(data_dir+'pubchem_pcba_merged.csv', low_memory=False)
merged_with_splits = merged.merge(smiles_df, on=['smiles'], how='outer')
print('Data shape before and after adding split column: ', merged.shape, merged_with_splits.shape)
print(pd.isnull(merged_with_splits['split']).any())
print(merged_with_splits.columns[-14])
merged_with_splits = merged_with_splits.sample(frac=1, random_state=42).reset_index(drop=True)
merged_with_splits.to_csv(data_dir+'pubchem_pcba_merged_splits.csv', header=True, index=False)


# Delete counter assay's actives
df = pd.read_csv(data_dir+'pubchem_pcba_merged_splits.csv', low_memory=True)
target_task = np.array(df['PCBA-2289'])
counter_task_1= np.array(df['PCBA-588342'])
counter_task_2= np.array(df['PCBA-411'])

countered_index = []
countered_counter = 0
for i in range(len(target_task)):
    if target_task[i] == 1:
        if counter_task_1[i] == 1 or counter_task_2[i] == 1:
            countered_index.append(i)
            countered_counter += 1
target_task_countered = target_task
for i in countered_index:
    target_task_countered[i] = np.NaN

df['PCBA-2289'] = np.array(target_task_countered)
df.to_csv(data_dir+'pubchem_pcba_merged_splits_countered.csv', header=True, index=False)
