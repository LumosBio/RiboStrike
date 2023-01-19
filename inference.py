import pandas as pd
import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models import GraphConvModel
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score, average_precision_score
from deepchem.metrics.score_function import bedroc_score
import time
import os
from rdkit.Chem import MolFromSmiles, MolToSmiles
import shutil
import logging
import itertools

logging.basicConfig(level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(123)
tf.compat.v1.set_random_seed(123)


def load_data(featurizer='GraphConv', split='scaffold', reload=True, delete_old_dataset=False, data_save_dir='./', data_dir=None, input_tasks=None):
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
        dummy_df = pd.read_csv(data_dir, low_memory=False)
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


def load_data_inference(featurizer='GraphConv', reload=True, delete_old_dataset=False, data_save_dir='./',
              data_dir=None, input_tasks=None):
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
    print("About to load the dataset.")

    # create featurizer, loader, transformers, and splitter
    if featurizer == 'ECFP':
        featurizer = dc.feat.CircularFingerprint(size=1024, chiral=True)
    elif featurizer == 'GraphConv':
        featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    loader = dc.data.InMemoryLoader(tasks=input_tasks, featurizer=featurizer)

    # check if built dataset exists on disk
    found_flag = 0
    if reload:
        loaded, dataset, transformers = dc.utils.data_utils.load_dataset_from_disk(data_save_dir)
        if loaded:
            test_dataset = dataset
            found_flag = 1
    # if the built dataset does not exist, create it
    if not found_flag:
        if not os.path.exists(dataset_file):
            print("Dataset not found")
        dummy_df = pd.read_csv(data_dir, low_memory=False, encoding= 'unicode_escape')
        smiles = np.array(dummy_df['smiles'])
        print("About to featurize the dataset.")
        train_dataset = loader.create_dataset([smiles[0]], shard_size=8192)
        valid_dataset = loader.create_dataset([smiles[0]], shard_size=8192)
        test_dataset = loader.create_dataset(smiles, shard_size=8192)
        transformers = [dc.trans.BalancingTransformer(dataset=test_dataset)]

        # dc.utils.data_utils.save_dataset_to_disk(data_save_dir, train=train_dataset, valid=valid_dataset,
        #                                          test=test_dataset, transformers=transformers)
    return tasks, (train_dataset, valid_dataset, test_dataset), transformers, loader


def plot_validation_roc(all_train_scores, all_valid_scores, fig_save_dir='results/roc_auc.png', title= 'Validation ROC AUC', markersize=10):
    fig = plt.figure(dpi=300)
    for variable in [all_train_scores, all_valid_scores]:
        plt.plot(range(1,len(variable)+1), variable, 'o-', markersize=markersize)
    plt.xlabel('Epoch', labelpad=0)
    plt.ylabel(title)
    plt.tight_layout()
    plt.savefig(fig_save_dir, format='png',dpi=300)
    plt.show()
    plt.close()


def plot_confusion_matrix(cm, fig_save_dir='results/untitled.png',
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = ['Inactive', 'Active']
    y_ticks_font = 11
    x_ticks_font = 11
    y_label_font = 13
    x_label_font = 13
    text_font = 11
    bar_font = 10
    plt.figure(figsize=(4, 3), dpi=300)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title, fontsize=20)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=bar_font)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=x_ticks_font)
    plt.yticks(tick_marks, classes, fontsize=y_ticks_font)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=text_font ,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=y_label_font)
    plt.xlabel('Predicted label', labelpad=0, fontsize=x_label_font)
    plt.tight_layout()
    plt.savefig(fig_save_dir, format='png', dpi=300)
    plt.show()


def canon_smile(smile):
    return MolToSmiles(MolFromSmiles(smile), isomericSmiles=True)

# run_type = 'single'
# run_type = 'multicancer'
# run_type = 'multi'
run_type= 'zeroshot'
# run_type= 'random'
# metric_valid = 'roc'
metric_valid = 'ap'
transfer = True
# transfer = False
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
epochnb = 150
if metric_valid == 'ap':
    if run_type == 'single':
        graph_conv_layers = [512, 512, 512]
        dropout = 0.1
        learning_rate = 0.0001
        batch_size = 128
        dense_layer_size = 1024
    elif run_type == 'multi' or run_type == 'multicancer':
        graph_conv_layers = [512, 512, 512]
        dropout = 0.1
        learning_rate = 0.0001
        batch_size = 128
        dense_layer_size = 1024
    elif run_type == 'zeroshot':
        graph_conv_layers = [512, 512, 512]
        dropout = 0
        learning_rate = 0.0001
        batch_size = 32
        dense_layer_size = 1024
elif metric_valid == 'roc':
    if run_type == 'multi':
        graph_conv_layers = [512, 512, 512]
        dropout = 0.1
        learning_rate = 0.0001
        batch_size = 128
        dense_layer_size = 1024
    elif run_type == 'single':
        graph_conv_layers = [512, 512, 512]
        dropout = 0.1
        learning_rate = 0.0001
        batch_size = 128
        dense_layer_size = 1024

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


metric = [
    dc.metrics.Metric(dc.metrics.accuracy_score, mode="classification", classification_handling_mode='threshold', threshold_value=0.5, n_tasks=len(input_tasks)),
    dc.metrics.Metric(dc.metrics.recall_score, mode="classification", classification_handling_mode='threshold', threshold_value=0.5, n_tasks=len(input_tasks)),
    dc.metrics.Metric(dc.metrics.roc_auc_score, mode="classification", n_tasks=len(input_tasks)),
    dc.metrics.Metric(dc.metrics.bedroc_score, mode="classification", n_tasks=len(input_tasks))]

model = None
model = GraphConvModel(
    len(all_tasks),
    batch_size=batch_size,
    mode='classification',
    number_atom_features=78,
    tensorboard=False,
    use_queue=True,
    graph_conv_layers=graph_conv_layers,
    dense_layer_size=dense_layer_size,
    dropout=dropout,
    learning_rate=learning_rate,
    model_dir=model_dir)


if transfer:
    if run_type == 'single':
        transfer_dir = 'built_models/mirna_cleaned/GraphConv/0618-1703/'
        transfer_epoch = 110
    elif run_type == 'multicancer':
        transfer_dir = 'built_models/mirna_cleaned/GraphConv/0619-1855/'
        transfer_epoch = 139
    elif run_type == 'multi':
        if metric_valid == 'ap':
            transfer_dir = 'built_models/mirna_cleaned/GraphConv/0619-1853/'
            transfer_epoch = 145
        elif metric_valid == 'roc':
            transfer_dir = 'built_models/mirna_cleaned/GraphConv/0628-0518/'
            transfer_epoch = int(round(147*225/150)/2)
    elif run_type == 'zeroshot':
        transfer_dir = 'built_models/mirna_cleaned/GraphConv/0628-0517/'
        transfer_epoch = 34
    elif run_type == 'random':
        transfer_dir = 'built_models/mirna_cleaned/GraphConv/0614-0207/'
        transfer_epoch = 86
    model.restore(checkpoint=transfer_dir+'ckpt-'+str(int(2*transfer_epoch)))

best_epoch = transfer_epoch

if run_type == 'single':
    y_pred_raw = model.predict(test_dataset)
    y_true = np.array(test_dataset.y)
    w = test_dataset.w
    w = np.reshape(w, (len(w)))
else:
    y_pred_raw = model.predict(test_dataset)[:,target_index]
    y_true = np.array(test_dataset.y[:, target_index])
    w = test_dataset.w[:, target_index]

y_pred_raw = np.reshape(y_pred_raw, (len(y_pred_raw),2))
y_pred = np.argmax(y_pred_raw, axis=1)
w_mask = w != 0
print(len(y_pred))
y_pred = y_pred[w_mask]
y_pred_raw = y_pred_raw[w_mask]
print(len(y_pred))
y_true = y_true[w_mask]
print(len(y_true))
print(np.sum(y_true))
cm = confusion_matrix(y_true, y_pred)
print(run_type)
print(model_dir, best_epoch)
print('ROC', roc_auc_score(y_true, y_pred_raw[:,1]), 'Accuracy', accuracy_score(y_true,y_pred),
      'Recall', recall_score(y_true, y_pred),
      'AP', average_precision_score(y_true, y_pred_raw[:,1]), 'Bedroc', bedroc_score(y_true, y_pred_raw))


# inference_type = 'zinc'
inference_type = 'asinex'


data_save_dir = 'built_datasets/'+inference_type+'/'+featurizer+'/' + run_type + '/' + metric_valid + '/'
data_dir = 'data/'+inference_type+'.csv'
reload = False
delete_old_dataset = False
if run_type != 'single':
    target_index_inf = target_index
else:
    target_index_inf = None
# Split dataset by scaffold into 80 10 10 percent splits
all_tasks, (train_dataset, valid_dataset, test_dataset), transformers, loader = load_data_inference(data_save_dir=data_save_dir,featurizer=featurizer, reload=reload, delete_old_dataset=delete_old_dataset, data_dir=data_dir, input_tasks=input_tasks)
training_data_len = len(test_dataset.y)


def predict_all_outputs(model, dataset, target_index_=None, return_logits=False, return_fings=False):
    generator = model.default_generator(
        dataset, mode='predict', deterministic=True, pad_batches=False)
    all_output_values = [[], [], []]
    batch_counter = 0
    output_counter = [0]
    if return_logits:
        output_counter.append(1)
    if return_fings:
        output_counter.append(2)
    for batch in generator:
        inputs, labels, weights = batch
        model._create_inputs(inputs)
        inputs, _, _ = model._prepare_batch((inputs, None, None))

        if len(inputs) == 1:
            inputs = inputs[0]
        output_values = model._compute_model(inputs)
        for i in output_counter:
            if target_index_ is None:
                all_output_values[i].extend(output_values[i])
            else:
                all_output_values[i].extend(output_values[i][:, target_index_])
        if batch_counter * batch_size % 10000 == 0:
            print(batch_counter * batch_size, len(dataset))
        batch_counter += 1
    return all_output_values


print(len(test_dataset))
outputs = predict_all_outputs(model, test_dataset, target_index_=target_index_inf, return_logits=True)
print(len(outputs), len(outputs[0]), len(outputs[1]))

preds = np.array(outputs[0])
logits = np.array(outputs[1])
# fingerprints = np.array(outputs[2])
del outputs
if run_type == 'single':
    preds = np.reshape(preds, (preds.shape[0], preds.shape[2]))
    logits = np.reshape(logits, (logits.shape[0], logits.shape[2]))

print(logits.shape)
print(preds.shape)

evidence = logits * (logits > 0)
alpha = evidence + 1
print(alpha.shape)
uncertainty = 2 / np.sum(alpha, axis=1)
uncertainty = uncertainty.flatten()


preds = np.argmax(preds, axis=1)
preds = preds.flatten()

print(preds[0], uncertainty[0])
dummy_df = pd.read_csv(data_dir, low_memory=False, encoding= 'unicode_escape')
dummy_df['uncertainty'] = uncertainty
dummy_df['pred'] = preds
dummy_df.to_csv('results/'+inference_type+'_'+run_type+'_'+metric_valid+'_preds.csv', header=True, index=False)




