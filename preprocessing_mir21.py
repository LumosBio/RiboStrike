import pandas as pd
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.compat.v1.set_random_seed(123)
import deepchem as dc
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
from rdkit.Chem import MolFromSmiles, MolToSmiles


def canon_smile(smile):
    return MolToSmiles(MolFromSmiles(smile), isomericSmiles=True)


# Folder that contains the dataset and will contain the preprocessed datasets
data_dir = 'data/pubchem/'

file_names = [
'193771_AID_1285_datatable_inhibit Alzheimer\'s amyloid precursor protein (APP) translation Alzheimer.csv',
'279988_AID_2675_datatable_type I myotonic dystrophy (DM1).csv',
'309031_AID_2551_datatable_all.csv',
'321735_AID_624417_datatable_all.csv',
'330308_AID_504466_datatable_ATAD5_ genotoxicity.csv',
'331108_AID_485367_datatable_Inhibitors of T. brucei phosphofructokinase.csv',
'335997_AID_623901_datatable_miR122 activator anticancer.csv',
'336623_AID_2289_datatable_Inhibitors of miR-21 anticancer.csv',
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
# file_names = [
# '336623_AID_2289_datatable_Inhibitors of miR-21 anticancer.csv']

# These are the source files, please keep the same
all_df = [pd.read_csv("/data/cleaned data/"+f, low_memory=False) for f in file_names]
for file_counter in range(len(all_df)):
    df = all_df[file_counter]
    print(file_names[file_counter])
    print('Initial shape: ', df.shape)
    print(df.columns)

    # Delete empty or duplicate smiles
    df = df.dropna(subset=['smiles'])
    df = df.drop_duplicates(subset='smiles', keep='first')
    df.reset_index(inplace=True)
    print('Shape after deleting empty or duplicate smiles: ', df.shape)

    # Find unique values in columns
    df_dropped = df.dropna()
    phenotype_column = None
    outcome_column = None
    for column in df.columns:
        if 'Phenotype' in column:
            phenotype_column = column
            print(column, np.unique(df_dropped[column]))
        if 'OUTCOME' in column:
            outcome_column = column
            print(column, np.unique(df_dropped[column]))
    if phenotype_column != None:
        print('Summary:')
        print(df.groupby(phenotype_column)['smiles'].nunique())

    # Delete activators
    if file_names[file_counter] in ['330308_AID_504466_datatable_ATAD5_ genotoxicity.csv', '377550_AID_624202_datatable_Activators of BRCA1 Expression anticancer.csv']:
        delete_type = 'Inhibitor'
    else:
        delete_type = 'Activator'

    if phenotype_column != None:
        df = df[df[phenotype_column] != delete_type]
        df = df[df[phenotype_column] != 'Cytotoxic']
        df = df[df[phenotype_column] != 'Fluorescent']
        df = df[df[phenotype_column] != 'Quencher']
    print('Shape after deleting activators (or inhibitors in some cases): ', df.shape)

    # Numericalize Labels
    df[outcome_column] = df[outcome_column].replace({'Active': 1, 'Inconclusive': 1, 'Inactive': 0, 'Unspecified':0})

    # Make all smiles canon
    if file_names[file_counter] == '309031_AID_2551_datatable_all.csv':
        df = df[df['smiles'] != 'F[Si-2](F)(F)(F)(F)F']
    smiles = df['smiles']
    smiles_canon = [canon_smile(s) for s in smiles]
    df['smiles'] = smiles_canon
    print('smiles and unique smiles: ', len(smiles_canon), len(np.unique(smiles_canon)))

    # Shuffle and save
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    all_df[file_counter] = df
    all_df[file_counter].to_csv(data_dir+ file_names[file_counter], header=True, index=False)

# Find overlap between smiles of all datasets with the target dataset
target_smiles = np.array(all_df[7]['smiles'])
for file_counter in range(len(all_df)):
    df = all_df[file_counter]
    current_smiles = np.array(df['smiles'])
    print(file_names[file_counter], 'smiles overlap percentage: ', len(np.intersect1d(target_smiles, current_smiles))/len(current_smiles)*100)

# Merge all dataframes using unique smiles
all_df = [pd.read_csv(data_dir+f, low_memory=False) for f in file_names]
for file_counter in range(len(all_df)):
    df = all_df[file_counter]
    outcome_column = None
    for column in df.columns:
        if 'OUTCOME' in column:
            outcome_column = column
            print(column, np.unique(df[column]))
    df = df[['smiles', outcome_column]]
    if file_counter == 0:
        merged = df
    else:
        merged = merged.merge(df, on=['smiles'], how='outer')
print(merged.head())
print(merged.shape)
print(merged.iloc[0])
smiles_canon = merged['smiles']
print('smiles and unique smiles: ', len(smiles_canon), len(np.unique(smiles_canon)))
print('number of rows with missing data: ', merged.shape[0] - merged.dropna().shape[0])
merged.to_csv(data_dir+'pubchem_merged.csv', header=True, index=False)

print(merged.columns)

input_data = data_dir+'pubchem_merged.csv'
input_columns = list(pd.read_csv(input_data, low_memory=True).columns)
input_tasks = list(np.array(input_columns)[[True if 'OUTCOME' in c else False for c in input_columns]])
print(input_tasks)
split = 'scaffold'
featurizer = 'GraphConv'

def load_mir21(featurizer='ECFP', split='random', input_data=input_data, input_tasks=input_tasks):
    # Load TOX21 dataset
    print("About to load the dataset.")
    dataset_file = input_data
    dataset = dc.utils.save.load_from_disk(dataset_file)
    print("Columns of dataset: %s" % str(dataset.columns.values))
    print("Number of examples in dataset: %s" % str(dataset.shape[0]))

    # Featurize TOX21 dataset
    print("About to featurize SR dataset.")

    if featurizer == 'ECFP':
        featurizer = dc.feat.CircularFingerprint(size=1024, chiral=True)
    elif featurizer == 'GraphConv':
        featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)

    mir21_tasks = input_tasks

    print(mir21_tasks,'is our task(s)')

    loader = dc.data.CSVLoader(tasks=mir21_tasks, smiles_field="smiles", featurizer=featurizer)

    dataset = loader.featurize(dataset_file, shard_size=8192)

    # Initialize transformers
    transformers = [dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]
    print("About to transform data")
    for transformer in transformers:
        dataset = transformer.transform(dataset)

    splitters = {
        'index': dc.splits.IndexSplitter(),
        'random': dc.splits.RandomSplitter(),
        'scaffold': dc.splits.ScaffoldSplitter()
    }
    splitter = splitters[split]

    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)

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
merged = pd.read_csv(data_dir+'pubchem_merged.csv')
merged_with_splits = merged.merge(smiles_df, on=['smiles'], how='outer')
print('Data shape before and after adding split column: ', merged.shape, merged_with_splits.shape)
print(pd.isnull(merged_with_splits['split']).any())
merged_with_splits.to_csv(data_dir+'pubchem_merged_splits.csv', header=True, index=False)


preprocess_inference = False
if preprocess_inference:
    df = pd.read_csv('data/zinc.csv')
    smiles_canon = [canon_smile(c) for c in np.array(df['smiles'])]
    df.rename(columns={'smiles': 'smiles_original'}, inplace=True)
    df['smiles'] = smiles_canon
    df.to_csv('data/zinc.csv', header=True, index=False)
