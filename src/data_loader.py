import pandas as pd


class LoanDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        loan = pd.read_csv(self.file_path)
        return loan

    