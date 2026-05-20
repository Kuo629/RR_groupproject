import numpy as np


class LoanCleaner:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def clean_outliers(self):
        self.df.loc[self.df["person_age"] > 100, "person_age"] = np.nan
        self.df.loc[self.df["person_income"] > 7200764, "person_income"] = np.nan
        self.df.loc[self.df["person_emp_exp"] > 124, "person_emp_exp"] = np.nan
        return self.df

    def mean_income_by_education(self):
        mean_income = (
            self.df
            .groupby("person_education")["person_income"]
            .mean()
            .reset_index()
            .sort_values(by="person_income", ascending=False)
        )
        print(mean_income)
        return mean_income

    def filter_age_income_education(self):
        filtered = self.df[self.df["person_age"].notnull()][
            ["person_age", "person_income", "person_education"]
        ]
        print(filtered.sort_values(by="person_income", ascending=False).head())
        return filtered