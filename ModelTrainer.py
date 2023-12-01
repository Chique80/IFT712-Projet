from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Ignorer les avertissements de division par zéro dans les métriques de classification
warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self, algorithms):
        self.algorithms = algorithms
        self.best_models = {}  # Dictionnaire pour stocker les meilleurs modèles de chaque algorithme

    def train_models(self, x_train, t_train, param_grids):
        for algo, param_grid in zip(self.algorithms, param_grids):
            grid_search = GridSearchCV(algo, param_grid, cv=5)
            grid_search.fit(x_train, t_train)
            self.best_models[algo.__class__.__name__] = {
                'best_model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_
            }

    def predict(self, x_test):
        predictions = {}
        for algo, best_model_info in self.best_models.items():
            best_model = best_model_info['best_model']
            predictions[algo] = best_model.predict(x_test)
        return predictions

    def evaluate_models(self, x_test, t_test):
        for algo, best_model_info in self.best_models.items():
            best_model = best_model_info['best_model']
            t_pred = best_model.predict(x_test)
            accuracy = accuracy_score(t_test, t_pred)
            report = classification_report(t_test, t_pred)
            
            print(f"Algorithm: {algo}")
            print(f"Best Hyperparameters: {best_model_info['best_params']}")
            print(f"Accuracy: {accuracy}")
            #print("Classification Report:\n", report)
            print("=" * 50)

    def get_best_models(self):
        return self.best_models
