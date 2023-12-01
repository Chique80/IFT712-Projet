import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def display_evaluation_reports(reports:pd.DataFrame) -> None:
    print('f1_score  : {:.4} ± {:.4}'.format(reports.f1_score.mean(), reports.f1_score.std()))
    print('precision : {:.4} ± {:.4}'.format(reports.precision.mean(), reports.precision.std()))
    print('recall    : {:.4} ± {:.4}'.format(reports.recall.mean(), reports.recall.std()))
    reports.boxplot(column=['f1_score', 'precision', 'recall'], figsize=(8,5), notch=1)
    pass

def evaluate_predictions(targets:list, predictions:list, labels:list, log:bool=True) -> (float, pd.DataFrame):
    """ Evaluate a list predictions using various metrics """
    acc = accuracy_score(targets, predictions)

    reports = pd.DataFrame(index=labels)

    score = f1_score(targets, predictions, average=None)
    reports['f1_score'] = score

    precision = precision_score(targets, predictions, average=None)
    reports['precision'] = precision

    recall = recall_score(targets, predictions, average=None)
    reports['recall'] = recall

    if log:
        print('Accuracy  : {:.4%}'.format(acc))
        display_evaluation_reports(reports)

    return acc, reports
    pass

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
        print("  - Nb classes   :", self.nb_classes, "( Dim :", len(self.data.columns), ")")
        print("  - Nb samples   :", len(self.data))
        print("     - Training  :", len(self.x_train), "(", len(self.x_train)/len(self.data)*100, "% )")
        print("     - Test      :", len(self.x_test), "(", len(self.x_test)/len(self.data)*100, "% )")
        print("########################")
        pass
