import json
import math
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self, model_path):
        self.load_model(model_path)
        self.features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    def load_model(self, model_path):
        
        with open(model_path, "r") as f:
            model_data = json.load(f)
            self.class_priors = model_data["class_priors"]
            self.class_params = model_data["class_params"]

    def gaussian_pdf(self, x, mean, var):
        
        exponent = math.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / (math.sqrt(2 * math.pi * var))) * exponent

    def predict(self, sample):
        
        log_probs = {}

        for label, params in self.class_params.items():
            log_prob = math.log(self.class_priors[label])

            for i in range(len(self.features)):
                log_prob += math.log(self.gaussian_pdf(sample[i], params["means"][i], params["vars"][i]))

            log_probs[label] = log_prob

        return max(log_probs, key=log_probs.get)
    
    def test(self, test_path): #This will be for evaluating the classifier with the test dataset.

        test_df = pd.read_csv(test_path)
        X_test = test_df.iloc[:, 1:5].values # This will give us the first four columns which are the features
        y_true = test_df.iloc[:, 5].values

        correct = 0

        total = len(y_true)

        for i in range(total):
            predicted_label = self.predict(X_test[i]) # X_test is basically the sample that we are inputting into the sample parameter
            actual_label = y_true[i] 

            print(f"Sample {i+1}: Predicted = {predicted_label}, Actual = {actual_label}")

            if predicted_label == actual_label: # comparing the predicted value with the actual value
                correct += 1
            
        accuracy = correct/total
        print(f"\nModel Accuracy: {accuracy:.2%}")



def main():
    classifier = NaiveBayesClassifier("model.json")

    classifier.test("./data/test.csv")

if __name__ == "__main__":
    main()
