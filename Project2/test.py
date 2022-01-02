
import pickle
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from train import extractFeaturesFromMealNoMealData


with open("model.pkl", 'rb') as file:
    model = pickle.load(file)
    df = pd.read_csv('test.csv', header=None)

# Extract test features
test_features = extractFeaturesFromMealNoMealData(df)

# normalize the test features
test_features = (test_features - test_features.mean(axis='index'))/ (test_features.max() - test_features.min())
fit = StandardScaler().fit_transform(test_features)
function_pca = PCA(n_components = 5)
fit_to_pca = function_pca.fit_transform(fit)

# Predict the values of test input
prediction = model.predict(fit_to_pca)

# Dump the values into Results.csv
pd.DataFrame(prediction).to_csv("Results.csv", header=None, index=False)