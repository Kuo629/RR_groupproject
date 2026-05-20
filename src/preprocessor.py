import pandas as pd
from sklearn.model_selection import train_test_split


class LoanPreprocessor:
    def __init__(self, dataframe, target_column="loan_status"):
        self.df = dataframe.copy()
        self.target_column = target_column

    def encode_features(self, drop_missing=False):
        if drop_missing:
            self.df = self.df.dropna().copy()

        if self.df[self.target_column].dtype == "object":
            self.df[self.target_column] = pd.factorize(self.df[self.target_column])[0]

        X = pd.get_dummies(
            self.df.drop(columns=[self.target_column]),
            drop_first=True
        )

        y = self.df[self.target_column]

        return X, y

    def split_data(self, X, y, test_size=0.25, random_state=1234):
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )