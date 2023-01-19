import pandas as pd
import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models import GraphConvModel
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score, average_precision_score, precision_score
from deepchem.metrics.score_function import bedroc_score
import time
import os
from rdkit.Chem import MolFromSmiles, MolToSmiles
import shutil
import logging
import itertools
import matplotlib

plt.rcParams["font.family"] = "Arial"

logging.basicConfig(level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    loader = dc.data.InMemoryLoader(tasks=["task1"], featurizer=featurizer)

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
        dummy_df = pd.read_csv(data_dir, low_memory=False)
        smiles = np.array(dummy_df['smiles'])
        print("About to featurize the dataset.")
        train_dataset = loader.create_dataset(smiles[0], shard_size=8192)
        valid_dataset = loader.create_dataset(smiles[0], shard_size=8192)
        test_dataset = loader.create_dataset(smiles, shard_size=8192)
        transformers = [dc.trans.BalancingTransformer(dataset=test_dataset)]

        dc.utils.data_utils.save_dataset_to_disk(data_save_dir, train=train_dataset, valid=valid_dataset,
                                                 test=test_dataset, transformers=transformers)
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
    y_ticks_font = 15
    x_ticks_font = 15
    y_label_font = 15
    x_label_font = 15
    text_font = 15
    bar_font = 15
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

# Select operation mode
# run_type = 'single'
# run_type = 'multicancer'
run_type = 'multi'
# run_type= 'zeroshot'
# run_type= 'random'

# Select validation metric
metric_valid = 'ap'

# Transfer from trained model, bypass training
transfer = True
data_dir = 'data/pubchem_pcba_merged_splits_countered.csv'
main_df = pd.read_csv(data_dir, low_memory=False)
input_columns = list(main_df.columns)
target_task = 'PCBA-2289'

# Select tasks based on the operation mode
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
    input_tasks = ['PCBA-881', 'PCBA-588579', 'PCBA-2517', 'PCBA-623901',
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
    elif run_type == 'zeroshot' or run_type == 'random':
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
    elif run_type == 'zeroshot_roc':
        graph_conv_layers = [512, 512, 512]
        dropout = 0
        learning_rate = 0.0001
        batch_size = 32
        dense_layer_size = 256
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

# Load pre-trained models
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
    elif run_type == 'zeroshot_roc':
        transfer_dir = 'built_models/mirna_cleaned/GraphConv/0701-2217/'
        transfer_epoch = 71
    elif run_type == 'random':
        transfer_dir = 'built_models/mirna_cleaned/GraphConv/1111-0813/'
        transfer_epoch = 139
    model.restore(checkpoint=transfer_dir+'ckpt-'+str(int(2*transfer_epoch)))


if not transfer:
    all_train_ap = []
    all_valid_ap = []
    all_train_roc = []
    all_valid_roc = []
    all_train_bedroc = []
    all_valid_bedroc = []
    all_loss = []
    epoch_number = 0
    best_valid_score = 0
    best_epoch = 0

    # Find non-empty datasets
    if run_type == 'single':
        w_mask_valid = list(np.reshape(np.array(valid_dataset.w != 0), len(np.array(valid_dataset.w != 0))))
        y_true_valid = np.array(valid_dataset.y)[w_mask_valid]
        w_mask_train = list(np.reshape(np.array(train_dataset.w != 0), len(np.array(train_dataset.w != 0))))
        y_true_train = np.array(train_dataset.y)[w_mask_train]
    else:
        w_mask_valid = valid_dataset.w[:, target_index] != 0
        y_true_valid = np.array(valid_dataset.y[:, target_index])[w_mask_valid]
        w_mask_train = train_dataset.w[:, target_index] != 0
        y_true_train = np.array(train_dataset.y[:, target_index])[w_mask_train]
    for epoch_num in range(epochnb):
        loss = model.fit(train_dataset, nb_epoch=1, checkpoint_interval=(training_data_len // batch_size),
                         max_checkpoints_to_keep=1000)
        all_loss.append(loss)

        print(epoch_num + 1, 'epoch number')
        print("Evaluating model")

        if run_type == 'single':
            y_pred_raw_train = model.predict(train_dataset)
            y_pred_raw_valid = model.predict(valid_dataset)

        else:
            y_pred_raw_train = model.predict(train_dataset)[:, target_index]
            y_pred_raw_valid = model.predict(valid_dataset)[:, target_index]

        y_pred_raw_train = np.reshape(y_pred_raw_train, (len(y_pred_raw_train), 2))[w_mask_train]
        y_pred_raw_valid = np.reshape(y_pred_raw_valid, (len(y_pred_raw_valid), 2))[w_mask_valid]
        all_train_roc.append(roc_auc_score(y_true_train, y_pred_raw_train[:, 1]))
        all_valid_roc.append(roc_auc_score(y_true_valid, y_pred_raw_valid[:, 1]))
        all_train_bedroc.append(bedroc_score(y_true_train, y_pred_raw_train))
        all_valid_bedroc.append(bedroc_score(y_true_valid, y_pred_raw_valid))
        all_train_ap.append(average_precision_score(y_true_train, y_pred_raw_train[:, 1]))
        all_valid_ap.append(average_precision_score(y_true_valid, y_pred_raw_valid[:, 1]))
        if metric_valid == 'ap':
            current_valid_score = all_valid_ap[-1]
        elif metric_valid == 'roc':
            current_valid_score = all_valid_roc[-1]
        print(current_valid_score)

        if current_valid_score > best_valid_score:
            best_valid_score = current_valid_score
            best_epoch = epoch_num + 1
            print('Better model found at epoch ', best_epoch, 'with validation score of ', best_valid_score)

    print(model_dir, best_epoch)
    results_df = pd.DataFrame(data={'all loss': all_loss, 'train roc': all_train_roc, 'valid roc': all_valid_roc,
                                    'train ap': all_train_ap, 'valid ap': all_valid_ap,
                                    'train bedroc': all_train_bedroc, 'valid bedroc': all_valid_bedroc,
                                    'model_dir': model_dir, 'best_epoch': best_epoch})
    results_df.to_csv(model_dir+'results.csv', header=True, index=False)

    # Load the best model
    model.restore(checkpoint=model_dir+'ckpt-'+str(int(2*best_epoch)))

    train_scores = model.evaluate(train_dataset, metric, transformers, per_task_metrics=True)[1]
    valid_scores = model.evaluate(valid_dataset, metric, transformers, per_task_metrics=True)[1]
    test_scores = model.evaluate(test_dataset, metric, transformers, per_task_metrics=True)[1]

    df_counter = 0
    df_names = ['train', 'valid', 'test']
    for score in [train_scores, valid_scores, test_scores]:
        df_name = df_names[df_counter]
        for score_type in ['accuracy_score', 'recall_score', 'roc_auc_score']:
            if run_type == 'single':
                print(df_name, score_type, score[score_type])
            else:
                print(df_name, score_type, score[score_type][target_index])
        df_counter += 1
else:
    best_epoch = transfer_epoch

# Evaluate on the non-empty test set
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
      'AP', average_precision_score(y_true, y_pred_raw[:,1]), 'Bedroc', bedroc_score(y_true, y_pred_raw),
      'Precision', precision_score(y_true, y_pred))

# Save a figure of train and valid rocauc scores and the confusion matrix
if not transfer:
    plot_validation_roc(all_train_roc, all_valid_roc, fig_save_dir=model_dir+'roc.png', title='Validation ROC AUC', markersize=4)
    plot_validation_roc(all_train_bedroc, all_valid_bedroc, fig_save_dir=model_dir+'bedroc.png', title='Validation BEDROC', markersize=4)
    plot_validation_roc(all_train_ap, all_valid_ap, fig_save_dir=model_dir+'ap.png', title='Validation Average Precision Score', markersize=4)
    plot_confusion_matrix(cm, fig_save_dir=model_dir+'confusion.png', normalize=True)


print(model_dir, best_epoch)

recommend_tasks = True
if recommend_tasks:
    # Find the performance of the different sub-models on the valdation set of the target task
    valid_zero = []
    y_pred_raw_all = model.predict(valid_dataset)
    y_true = np.array(valid_dataset.y[:, target_index])
    w = valid_dataset.w[:, target_index]
    w_mask = w != 0
    y_true = y_true[w_mask]
    for t in range(len(input_tasks)):
        y_pred_raw = y_pred_raw_all[:,t]
        y_pred_raw = np.reshape(y_pred_raw, (len(y_pred_raw),2))
        y_pred = np.argmax(y_pred_raw, axis=1)
        print(len(y_pred))
        y_pred = y_pred[w_mask]
        y_pred_raw = y_pred_raw[w_mask]
        print(len(y_pred))
        print(len(y_true))
        print(np.sum(y_true))
        cm = confusion_matrix(y_true, y_pred)
        print(roc_auc_score(y_true, y_pred_raw[:,1]), accuracy_score(y_true,y_pred),recall_score(y_true, y_pred),
              average_precision_score(y_true, y_pred_raw[:,1]))
        if metric_valid == 'roc':
            valid_zero.append(roc_auc_score(y_true, y_pred_raw[:,1]))
        elif metric_valid == 'ap':
            valid_zero.append(average_precision_score(y_true, y_pred_raw[:,1]))


    # Find the top performing models on the validation set and choose their related task
    threshold = np.mean(valid_zero)+2*np.std(valid_zero)
    task_mask = valid_zero >= threshold
    selected_tasks = np.array(input_tasks)[task_mask]
    print(selected_tasks)
    selected_tasks_names = [k.lstrip('PCBA-') for k in selected_tasks]

    display_task_mask = task_mask.copy()
    display_task_mask[target_index] = False

    fontsize = 14
    width = 0.8
    fig = plt.figure(dpi=300)
    # plt.plot(range(1,len(valid_zero)+1), valid_zero, 'o-')
    # plt.bar(input_tasks, valid_zero, width=0.4)
    plt.bar(np.array(range(1, len(valid_zero) + 1))[target_index], np.array(valid_zero)[target_index], width=width,
            label='Target Task (miRNA)', color=matplotlib.colormaps['Set3'](3))
    plt.bar(np.array(range(1, len(valid_zero) + 1))[display_task_mask], np.array(valid_zero)[display_task_mask], width=width,
            label='Recommended Tasks', color=matplotlib.colormaps['Set3'](0))
    plt.bar(np.array(range(1, len(valid_zero) + 1))[~task_mask], np.array(valid_zero)[~task_mask], width=width,
            label='Other Tasks', color=matplotlib.colormaps['Set3'](10))
    plt.hlines(y=np.mean(valid_zero)+2*np.std(valid_zero), xmin=0, xmax=140, linestyles='dashed', color='black', label='Threshold')
    # plt.bar_label(p1)
    # plt.xticks(np.array(range(1, len(valid_zero) + 1))[task_mask], selected_tasks, rotation=90, fontsize=fontsize-5)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Task (Assay) Number', labelpad=0, fontsize=fontsize)
    plt.ylabel('Target Validation Average Precision', fontsize=fontsize)
    plt.xlim((-2,142))
    plt.legend(fontsize=fontsize-2, fancybox=True)
    plt.tight_layout()
    # plt.savefig(model_dir+'zero_shot.png', format='png', dpi=300)
    plt.savefig(model_dir+'zero_shot_general.png', format='png', dpi=300)
    plt.show()
    plt.close()
