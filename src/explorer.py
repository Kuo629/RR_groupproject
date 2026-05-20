import numpy as np


class LoanExplorer:
    def __init__(self, dataframe):
        self.df = dataframe

    def show_basic_info(self):
        print(self.df.head(10))
        print("Data dimensions:", self.df.shape)

        print("\n--- Data Info ---")
        self.df.info()

        print("\n--- Summary Statistics ---")
        print(self.df.describe(include="all"))

    def show_numeric_summary(self):
        print("\n--- Numeric Summary ---")
        print(self.df.select_dtypes(include=[np.number]).describe())

    def show_categorical_summary(self):
        print("\n--- Categorical Summary ---")
        print(self.df.select_dtypes(include=["object", "category"]).describe())

    def show_missing_values(self):
        print("\n--- Missing values per column ---")
        print(self.df.isnull().sum())
        print("\nTotal missing values:", self.df.isnull().sum().sum())