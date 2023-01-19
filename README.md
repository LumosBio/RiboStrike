# RiboStrike
RiboStrike successfully performs virtual screening for miR-21. State-of-the-art deep learning techniques such as multitask learning and hyper-parameter optimization through external validation are leveraged to create a practical and accurate pipeline in silico.

# Prerequisites
deepchem==2.5.0

scikit-learn==0.24.2

tensorflow-gpu==2.5.0 

# Installation
Install Deepchem from its github repository https://github.com/deepchem/deepchem. Installation in a conda environment is recommended. Having done so, clone this repository.

# Preprocessing
The preprocessing step canonicalized the SMILES, deletes duplicates or missing SMILES, deletes activators (or inhibitors in some cases where they are the minority), numericalizes the label (active and inconclusive are 1 in most cases), and shuffles the data. In toxicity data, this step also desalts the molecules and removes inorganic molecules. After preprocessing_mir21, preprocessing_merge_pcba will combine our data with the PCBA dataset, deletes overlapping tasks from the PCBA dataset, and splits the data into train, test, and validation sets based on molecular scaffold. In the end, the conter tasks' positives are also removed from the main miR-21 task data.

# Hyper-Parameter Optimization
During hyper-opt the dataset is loaded and based on the mode of operation (singletask, multitask, or zeroshot) different tasks are selected from the dataset. Th selected data are then featurized and modeled using a dictionary of hyper-parameters. The results are saved as a CSV file, with the most successful model being found using validation AP score.

# Training
In order to train the model, the operation mode is selected and the selected data is featurized and saved. The optimum hyper-parameters from the last step are used to create the model, which is then trained on the training set. Validation set is used to find the best epoch during training, and the best model's weights are loaded in the end. The model is then evaluated on the test set. 

# Inference
During inference "transfer" is set to True to load the pre-traiend model. The inference dataset type (zinc, lincs, asinex) is set and the related data is loaded and featurized. The predictions and their uncertainties are calculated and saved.

# Task Recommendation
Once the multitask model is trained on all tasks (at the end of the training code), the sub-models for all tasks are evaluated on the validation set of the target task. By doing so, the models which align the most with the target task's model are found. The tasks relating to these sub-models are chosen and are used as the input for the 'zeroshot' model.

# Molecule Selection
In the clustering code, similar to inference, the trained model is loaded alongside the inference datasets. The inner representations of the model from the last layer are extracted as fingerprints for each molecule. The fingerprints for top predictions of zinc, lincs, and FDA dataset as well as the fingerprints from the training dataset are clustered using Kmeans clustering (k=10) on top of their UMAP. After clusters are defined, top ten molecules (least uncertainties) are selected from zinc for each cluster. The selected molecules are then passed to the toxicity and dicer models for inference.
