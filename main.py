from src.data_loader import LoanDataLoader
from src.explorer import LoanExplorer
from src.cleaner import LoanCleaner
from src.visualizer import LoanVisualizer
from src.preprocessor import LoanPreprocessor
from src.models import DecisionTreeLoanModel, RandomForestLoanModel


def main():
    loader = LoanDataLoader("data/loan_data.csv")
    loan = loader.load_data()

    explorer = LoanExplorer(loan)
    explorer.show_basic_info()
    explorer.show_numeric_summary()
    explorer.show_categorical_summary()
    explorer.show_missing_values()

    visualizer = LoanVisualizer(loan)
    visualizer.plot_all()

    cleaner = LoanCleaner(loan)
    cleaned_loan = cleaner.clean_outliers()
    cleaner.mean_income_by_education()
    cleaner.filter_age_income_education()

    # Decision Tree
    preprocessor_dt = LoanPreprocessor(cleaned_loan)
    X, y = preprocessor_dt.encode_features(drop_missing=False)
    X_train, X_test, y_train, y_test = preprocessor_dt.split_data(X, y)

    dt_model = DecisionTreeLoanModel()
    dt_model.train(X_train, y_train)
    dt_accuracy = dt_model.evaluate(X_test, y_test)
    dt_model.plot_tree_model(X.columns)

    # Random Forest
    preprocessor_rf = LoanPreprocessor(cleaned_loan)
    X_rf, y_rf = preprocessor_rf.encode_features(drop_missing=True)
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = preprocessor_rf.split_data(X_rf, y_rf)

    rf_model = RandomForestLoanModel()
    rf_model.train(X_train_rf, y_train_rf)
    rf_accuracy = rf_model.evaluate(X_train_rf, y_train_rf, X_test_rf, y_test_rf)
    rf_model.plot_feature_importance(X_rf.columns)

    print("\n--- Final Analysis Conclusion ---")
    print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

    if rf_accuracy > dt_accuracy:
        print("Random Forest performed better than Decision Tree.")
    else:
        print("Decision Tree performed better than Random Forest.")


if __name__ == "__main__":
    main()