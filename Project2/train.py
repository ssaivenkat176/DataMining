from numpy.core.fromnumeric import argmax, argmin
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import pickle
from scipy.fftpack import fft
import pandas as pd
import numpy as np


def capture_data_meal_no_meal(cgm_data, insulin_data, is_meal) :
    '''
    The function takes in combined cgm data and insulin data and extracts the meal data or the no meal
    data based on the boolean field is_meal
    :param cgm_data: Parameter indicting cgm glucose data
    :param insulin_data: Parameter indicting insulin data
    :param is_meal: indicator which indicates whether we are capturing meal data or no meal data
    :return: meal data or no meal data
    '''

    time_of_data, meal_no_meal_data = [],[]
    if is_meal is True:
        reqMinuteDur = 120
        reqLength = 30
    else:
        reqMinuteDur = 240
        reqLength = 24
    # Loop over the insulin data
    for i in range(len(insulin_data['date_time']) - 1):
        if (insulin_data['date_time'].iloc[i + 1] - insulin_data['date_time'].iloc[i]).seconds // 60 < reqMinuteDur:
            continue
        time_of_data.append(insulin_data['date_time'].iloc[i])

    if is_meal == True:
        for date_time in time_of_data:

            start = date_time - pd.Timedelta(minutes = 30)
            end = date_time + pd.Timedelta(minutes = 120)

            cgm_data_date_time_stretch = \
            cgm_data.loc[(cgm_data['date_time'] >= start) & (cgm_data['date_time'] <= end)][
                'Sensor Glucose (mg/dL)'].values.tolist()

            if len(cgm_data_date_time_stretch) < reqLength:
                continue
            cgm_data_date_time_stretch = np.asarray(cgm_data_date_time_stretch)
            meal_no_meal_data.append(cgm_data_date_time_stretch[:reqLength])
    else:
        for i in range(len(time_of_data) - 1):
            start = time_of_data[i] + pd.Timedelta(minutes = 120)
            end = time_of_data[i + 1]
            cgm_data_stretch = cgm_data.loc[(cgm_data['date_time'] >= start) & (cgm_data['date_time'] <= end)][
                'Sensor Glucose (mg/dL)'].values.tolist()

            if len(cgm_data_stretch) < reqLength:
                continue

            meal_no_meal_data.append(cgm_data_stretch[:reqLength])

    return pd.DataFrame(meal_no_meal_data)

def calculate_max_zero_crossings(feature_row, highestValueIndex):

    # Calculate slopes for each x value
    slopes = np.diff(feature_row)
    zero_cross = []
    initSign = 1 if slopes[0] > 0 else 0

    # Check where slope sign changes
    for x in range(1, len(slopes)):
        newSign = 1 if slopes[x] > 0 else 0
        if initSign != newSign:
            # Append to zero_cross list
            zero_cross.append([slopes[x] - slopes[x-1], x])
        initSign = newSign

    if highestValueIndex < len(zero_cross):
        return sorted(zero_cross, reverse=True)[highestValueIndex]
    else:
        return [0, 0]



def calculate_fast_fourier(value, index):

    fastFourier = np.abs(fft(value))
    sortedAmplitude = sorted(fastFourier, reverse=True)
    maximum_amplitude = sortedAmplitude[index]
    return maximum_amplitude


def extractFeaturesFromMealNoMealData(data):
    '''

    :param data: meal or no meal data from which features need to be calculated
    :return: return all the features extracted from the input data
    '''

    features = pd.DataFrame()
    dict = {}
    for index in range(0, data.shape[0]):
        value = data.iloc[index, :].values.tolist()
        dict['MaxMinFeature'] = max(value) - min(value)
        dict['ArgMaxMinFeature'] = argmax(value) - argmin(value)

        dict['ZeroCrossingMax1'] = calculate_max_zero_crossings(value , 0)[0]
        dict['ZeroCrossingMaxIndex1'] = calculate_max_zero_crossings(value, 0)[1]
        dict['ZeroCrossingMax2'] = calculate_max_zero_crossings(value, 1)[0]
        dict['ZeroCrossingMaxIndex2'] = calculate_max_zero_crossings(value, 1)[1]
        dict['ZeroCrossingMax3'] = calculate_max_zero_crossings(value, 2)[0]
        dict['ZeroCrossingMaxIndex3'] = calculate_max_zero_crossings(value, 2)[1]

        dict['MaxFFTAmp1'] = calculate_fast_fourier(value, 1)
        dict['MaxFFTAmp2'] = calculate_fast_fourier(value, 2)
        dict['MaxFFTAmp3'] = calculate_fast_fourier(value, 3)
        # Append all the features into features data frame
        features = features.append(dict, ignore_index=True)

    return features

if __name__ == "__main__":

    # Capture data of patient 1 from CGMData.csv and InsulinData.csv and filter out the columns which are not required
    cgm_data = pd.read_csv('CGMData.csv', low_memory=False, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
    insulin_data = pd.read_csv('InsulinData.csv', low_memory=False, usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])

    # Capture data of patient 2 and filter out the columns which are not required
    insulin_data_patient = pd.read_csv("Insulin_patient2.csv",
                                       usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])

    cgm_data_patient = pd.read_csv("CGM_patient2.csv",
                                   usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
    # Combine the cgm data from both patients
    cgm_data_df = pd.concat([cgm_data, cgm_data_patient])

    # Combine the insulin data from both patients
    insulin_data_df = pd.concat([insulin_data, insulin_data_patient])

    # Create a date_time field in cgm data by combining the data and time
    cgm_data_df['date_time'] = pd.to_datetime(
        cgm_data_df['Date'].astype(str) + " " + cgm_data_df['Time'].astype(str))

    # Create a date_time field in insulin data by combining the data and time

    insulin_data_df['date_time'] = pd.to_datetime(
        insulin_data_df['Date'].astype(str) + " " + insulin_data_df['Time'].astype(str))

    # Sort the values of the insulin data
    insulin_data_df = insulin_data_df.sort_values(by='date_time')

    # Filter out the rows where the column BWZ Carb Input (grams) in insulin values in NA or 0
    insulin_data_df = insulin_data_df[(insulin_data_df['BWZ Carb Input (grams)'].notna()) & (insulin_data_df['BWZ Carb Input (grams)'] > 0.0)]

    # Using interpolation to handling missing values in cgm data
    cgm_data['Sensor Glucose (mg/dL)'] = cgm_data['Sensor Glucose (mg/dL)'].interpolate(method='linear',
                                                                                        limit_direction='both')
    # Extract meal data
    meal_data = capture_data_meal_no_meal(cgm_data_df, insulin_data_df, True)

    # Extract no meal data
    no_meal_data = capture_data_meal_no_meal(cgm_data_df, insulin_data_df, False)

    meal_data = meal_data.dropna()
    no_meal_data = no_meal_data.dropna()

    # Extract all the required features from meal data
    mealDataFeatures = extractFeaturesFromMealNoMealData(meal_data)

    # Extract all the required features from no meal data
    noMealDataFeatures = extractFeaturesFromMealNoMealData(no_meal_data)

    # Normalizing the features of meal data and no meal data
    mealDataFeatures = (mealDataFeatures - mealDataFeatures.mean(axis='index')) / (mealDataFeatures.max() - mealDataFeatures.min())
    noMealDataFeatures = (noMealDataFeatures - noMealDataFeatures.mean(axis='index')) / (noMealDataFeatures.max() - noMealDataFeatures.min())

    stdScaler = StandardScaler()
    meal_std = stdScaler.fit_transform(mealDataFeatures)
    noMeal_std = stdScaler.fit_transform(noMealDataFeatures)

    # Defining the principal component analysis with 5 components
    # The number of components is chosen after trying different values like 8, 10
    function_pca = PCA(n_components = 5)

    meal_final_features = pd.DataFrame(function_pca.fit_transform(meal_std))
    # Set the class 1 to all the meal data
    meal_final_features['class'] = 1

    noMeal_final_features = pd.DataFrame(function_pca.fit_transform(noMeal_std))
    # Set the class 0 to all the meal data

    noMeal_final_features['class'] = 0

    combine_data = meal_final_features.append(noMeal_final_features)

    training_data,training_labels = combine_data.iloc[:, :-1] , combine_data.iloc[:, -1]

    kfold = KFold(5, True, 1)

    for x, y in kfold.split(training_data, training_labels):
        X_train, X_test = training_data.iloc[x], training_data.iloc[y]
        Y_train, Y_test = training_labels.iloc[x], training_labels.iloc[y]

        # Using Support Vector machine with RBF kernel
        model = SVC(kernel='rbf', gamma='scale', degree=3)
        model.fit(X_train, Y_train)

    with open('model.pkl', 'wb') as (modelfile):
        pickle.dump(model, modelfile)