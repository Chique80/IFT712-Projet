from DataManagement import Dataset
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from ModelTrainer import ModelTrainer


# Chemin vers le fichier CSV
csv_filepath = "data/train.csv"

# Création de l'objet Dataset
dataset = Dataset(csv_filepath)

# Division des données en ensemble d'entraînement et ensemble de test
x_train, t_train, x_test, t_test = dataset.split_data(test_size=0.2, seed=42, stratified=True)


# Liste des algorithmes à tester
algorithms = [
    RandomForestClassifier(),
    AdaBoostClassifier(),
    SVC(),
    KNeighborsClassifier(),
    LogisticRegression(),
    DecisionTreeClassifier()
]

# Listes des grilles d'hyperparamètres correspondantes
param_grids = [
    {
        # Hyperparamètres RandomForestClassifier
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
    },
    {
        # Hyperparamètres AdaBoostClassifier
        'n_estimators': [10, 50, 100],
        'learning_rate': [0.01, 0.1, 1.0],
    },
    {
        # Hyperparamètres SVC
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
    },
    {
        # Hyperparamètres KNeighborsClassifier
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
    },
    {
        # Hyperparamètres LogisticRegression
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2','elasticnet'],
    },
    {
        # Hyperparamètres DecisionTreeClassifier
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }
]

# Création de l'objet ModelTrainer avec les algorithmes
model_trainer = ModelTrainer(algorithms)

# Entraînement des modèles avec recherche d'hyperparamètres
model_trainer.train_models(x_train, t_train, param_grids)

# Évaluation des modèles
model_trainer.evaluate_models(x_test, t_test)

# Obtention des meilleurs modèles
best_models = model_trainer.get_best_models()
