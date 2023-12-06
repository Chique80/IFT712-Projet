import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score

def plot_classes_distribution(classes:list, samples:list) -> None:
    """ Plots the number of sample in each classes """
    nb_classes = len(classes)
    counts = np.zeros(nb_classes)

    for i in range(0, nb_classes):
        counts[i] = np.sum(samples == i)
    avg = counts.mean()
    std = counts.std()

    # Plotting
    plt.figure(figsize=(10,8))

    plt.bar(classes, counts, width=1.5)
    plt.axhline(avg, color = 'black', linewidth = 2)
    plt.axhline(avg+std, color = 'green', linewidth = 2)
    plt.axhline(avg-std, color = 'green', linewidth = 2)
    plt.xlabel("Classes")
    plt.ylabel("Nombre de données")

    plt.show()
    pass

class Dataset:
    """ Class to hold and manage the data
        
        Variables:
            filepath:   path of the csv file containing the data
            data:       data loaded from the csv as a DataFrame
            classes:    list of all the classes in the dataset
            nb_classes: nb of classes (length of 'classes')

            x_train:    list of the samples in the training set
            t_train:    list of the label of each sample in the training set
            id_train:   list of the id of each sample in the training set

            x_test:    list of the samples in the testing set
            t_test:    list of the label of each sample in the testing set
            id_test:   list of the id of each sample in the testing set
    """
    def __init__(self, filepath:str) -> None:
        self.filepath = filepath
        self.__load_data__()
        pass

    def __load_data__(self) -> None:
        self.data = pd.read_csv(self.filepath)

        # Convert species name to int
        le = LabelEncoder().fit(self.data.species)
        self.data.insert(2, 'label', le.transform(self.data.species), True)

        # Save a list of classes
        self.classes = le.classes_
        self.nb_classes = len(self.classes)
        pass

    def split_data(self, test_size:float, seed:int=0, stratified:bool=False) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        samples = self.data.drop(['id', 'species', 'label'], axis=1).to_numpy(copy=True)
        labels = self.data['label'].to_numpy(copy=True)
        ids = self.data['id'].to_numpy(copy=True)

        if stratified:
            sets = train_test_split(samples, labels, ids, test_size=test_size, random_state=seed, shuffle=True, stratify=labels)
        else:
            sets = train_test_split(samples, labels, ids, test_size=test_size, random_state=seed, shuffle=True)
        
        self.x_train = sets[0]
        self.t_train = sets[2]
        self.id_train = sets[4]

        self.x_test = sets[1]
        self.t_test = sets[3]
        self.id_test = sets[5]
        print(self.x_train.shape,self.x_test.shape)


        return self.x_train, self.t_train, self.x_test, self.t_test
        pass

    def plot_classes_distribution(self) -> None:
        class_labels = np.arange(0, self.nb_classes)
        plot_classes_distribution(class_labels, self.t_train)
        plot_classes_distribution(class_labels, self.t_test)
        pass

    def info(self) -> None:
        print("##### Informations #####")
        print("  - Nb classes   :", self.nb_classes, "( Dim :", len(self.data.columns)-3, ")")
        print("  - Nb samples   :", len(self.data))
        print("     - Training  :", len(self.x_train), "(", len(self.x_train)/len(self.data)*100, "% )")
        print("     - Test      :", len(self.x_test), "(", len(self.x_test)/len(self.data)*100, "% )")
        print("########################")
        pass


def train_model(model, hyperparameters:dict, dataset:Dataset) -> (GridSearchCV, pd.DataFrame):
    """ Search for the best hyperparameters for the given model and return the results """
    # Use 5-fold cross validation to find the best hyperparameters
    grid = GridSearchCV(model, hyperparameters, refit=True, verbose = 1, cv=5)
    grid.fit(dataset.x_train, dataset.t_train)

    # Format the results into a DataFrame for easy analysis
    columns = ['param_'+param for param in hyperparameters.keys()]
    columns.append('mean_test_score')
    columns.append('rank_test_score')
    results = pd.DataFrame(grid.cv_results_, columns=columns)

    return grid, results
    pass


def display_evaluation_reports(reports:pd.DataFrame) -> None:
    print('### Averages ###')
    print('   - precision : {:.3%} ± {:.3%}'.format(reports.precision.mean(), reports.precision.std()))
    print('   - recall    : {:.3%} ± {:.3%}'.format(reports.recall.mean(), reports.recall.std()))
    print('   - f1_score  : {:.3%} ± {:.3%}'.format(reports.f1_score.mean(), reports.f1_score.std()))
    print('################')
    reports.boxplot(column=['precision', 'recall', 'f1_score'], figsize=(8,5))
    pass


def evaluate_predictions(targets:list, predictions:list, labels:list, log:bool=True) -> pd.DataFrame:
    """ Evaluate a list predictions using various metrics """
    reports = pd.DataFrame(index=labels)

    precision = precision_score(targets, predictions, average=None)
    reports['precision'] = precision

    recall = recall_score(targets, predictions, average=None)
    reports['recall'] = recall

    score = f1_score(targets, predictions, average=None)
    reports['f1_score'] = score

    if log:
        display_evaluation_reports(reports)

    return reports
    pass

def extract_bad_predictions(targets:list, predictions:list, labels:list=None, ids:list=None) -> pd.DataFrame:
    """ Extract all the wrong predictions and place them inside a dataframe """
    bad_preds = targets != predictions      # Mask for bad predictions

    bad_preds_true = targets[bad_preds]
    bad_preds_predicted = predictions[bad_preds]

    df = pd.DataFrame()

    if labels is None:
        df['True Class'] = bad_preds_true
        df['Predicted Class'] = bad_preds_predicted
    else:
        df['True Class'] = labels[bad_preds_true]
        df['Predicted Class'] = labels[bad_preds_predicted]
    
    if ids is not None:
        df['Sample ID'] = ids[bad_preds]

    return df
    pass