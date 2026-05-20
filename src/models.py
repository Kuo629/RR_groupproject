import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class DecisionTreeLoanModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(
            criterion="entropy",
            random_state=1234,
            max_depth=4
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)

        print("\n--- Decision Tree Confusion Matrix ---")
        print(confusion_matrix(y_test, predictions))

        accuracy = accuracy_score(y_test, predictions)
        print(f"Decision Tree Accuracy: {accuracy:.4f}")

        print("\n--- Classification Report ---")
        print(classification_report(y_test, predictions))

        return accuracy

    def plot_tree_model(self, feature_names):
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            feature_names=feature_names,
            class_names=True,
            filled=True,
            fontsize=10
        )
        plt.title("Decision Tree Structure")
        plt.savefig("outputs/figures/decision_tree.png")
        plt.close()


class RandomForestLoanModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=2001,
            max_features=4,
            random_state=1234,
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_train, y_train, X_test, y_test):
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)

        print("\n--- RF Train Confusion Matrix ---")
        print(confusion_matrix(y_train, train_predictions))
        print("Train Accuracy:", accuracy_score(y_train, train_predictions))

        print("\n--- RF Test Confusion Matrix ---")
        print(confusion_matrix(y_test, test_predictions))

        test_accuracy = accuracy_score(y_test, test_predictions)
        print(f"Random Forest Test Accuracy: {test_accuracy:.4f}")

        return test_accuracy

    def plot_feature_importance(self, feature_names):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(10, 8))
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.savefig("outputs/figures/random_forest_feature_importance.png")
        plt.close()