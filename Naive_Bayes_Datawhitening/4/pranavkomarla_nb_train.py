import pandas as pd
import numpy as np
import json

class NaiveBayesTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.class_priors = {}
        self.class_params = {}
        self.features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    def load_data(self):
        
        self.df = pd.read_csv(self.data_path)
        self.class_labels = self.df["species"].unique()

    def train(self):
        
        for label in self.class_labels:
            subset = self.df[self.df["species"] == label]
            self.class_priors[label] = len(subset) / len(self.df)

            means = subset[self.features].mean().to_list()
            variances = subset[self.features].var(ddof=1).to_list()

            self.class_params[label] = {"means": means, "vars": variances}

    def save_model(self, filename="model.json"):
        
        model_data = {
            "class_priors": self.class_priors,
            "class_params": self.class_params
        }
        with open(filename, "w") as f:
            json.dump(model_data, f)
        print(f"Model saved to {filename}")

def main():
    trainer = NaiveBayesTrainer("./data/train.csv")
    trainer.load_data()
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
