import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class LoanVisualizer:
    def __init__(self, dataframe):
        self.df = dataframe
        sns.set_theme(style="whitegrid")
        os.makedirs("outputs/figures", exist_ok=True)

    def plot_age_by_education(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=self.df,
            x="person_age",
            hue="person_education",
            element="step",
            stat="count",
            common_norm=False
        )
        plt.title("Histogram of Age by Education")
        plt.savefig("outputs/figures/age_by_education.png")
        plt.close()

    def plot_numeric_histograms(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols].hist(figsize=(12, 10), bins=20)
        plt.suptitle("Histograms for all Numeric Data")
        plt.savefig("outputs/figures/numeric_histograms.png")
        plt.close()

    def plot_income_by_home_ownership_gender(self):
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=self.df,
            x="person_home_ownership",
            y="person_income",
            hue="person_gender",
            errorbar=None
        )
        plt.title("Income by Home Ownership and Gender")
        plt.savefig("outputs/figures/income_home_ownership_gender.png")
        plt.close()

    def plot_loan_intent(self):
        order = self.df["loan_intent"].value_counts().index

        plt.figure(figsize=(10, 5))
        sns.countplot(
            data=self.df,
            x="loan_intent",
            order=order,
            hue="loan_intent",
            legend=False
        )
        plt.title("Count for Loan Intent")
        plt.xticks(rotation=45)
        plt.savefig("outputs/figures/loan_intent_count.png")
        plt.close()

    def plot_all(self):
        self.plot_age_by_education()
        self.plot_numeric_histograms()
        self.plot_income_by_home_ownership_gender()
        self.plot_loan_intent()