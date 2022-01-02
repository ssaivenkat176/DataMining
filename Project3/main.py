from math import ceil, log
from numpy.core.fromnumeric import argmax
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN


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
    # sort the rows by date time stamp
    insulin_data = insulin_data.sort_values(by='date_time')
    # filter the column 'BWZ Carb Input (grams)' for non NAN non zero values.
    insulin_data = insulin_data[insulin_data['BWZ Carb Input (grams)'] != 0.0].dropna()

    insulin_data = insulin_data.reset_index().drop(columns='index')
    # interpolate for missing data
    cgm_data['Sensor Glucose (mg/dL)'] = cgm_data['Sensor Glucose (mg/dL)'].interpolate(method='linear',
                                                                                        limit_direction='both')

    if is_meal is True:
        reqMinuteDur = 120
        reqLength = 30
    else:
        reqMinuteDur = 240
        reqLength = 24

    # Loop over the insulin data
    carbs = []
    for i in range(len(insulin_data['date_time']) - 1):
        if (insulin_data['date_time'].iloc[i + 1] - insulin_data['date_time'].iloc[i]).seconds // 60 < reqMinuteDur:
            continue
        time_of_data.append(insulin_data['date_time'].iloc[i])
        carbs.append(insulin_data['BWZ Carb Input (grams)'][i])

    min_carb, max_carb = min(carbs), max(carbs)
    no_of_bins = ceil((max_carb - min_carb) / 20)

    meal_data = []
    ground_truth = []

    if is_meal == True:
        for i in range(len(time_of_data)):
            start = time_of_data[i] - pd.Timedelta(minutes = 30)
            end = time_of_data[i]  + pd.Timedelta(minutes = 120)

            cgm_data_date_time_stretch = cgm_data.loc[(cgm_data['date_time'] >= start) & (cgm_data['date_time'] <= end)][
                'Sensor Glucose (mg/dL)'].values.tolist()

            if len(cgm_data_date_time_stretch) < reqLength:
                continue
            meal_data.append(cgm_data_date_time_stretch[:30])
            ground_truth.append(int(ceil((carbs[i] - min_carb) / 20))) if carbs[
                                                                                  i] != min_carb else ground_truth.append(
                    1)

    else:
        for i in range(len(time_of_data) - 1):
            start = time_of_data[i] + pd.Timedelta(minutes = 120)
            end = time_of_data[i + 1]
            cgm_data_stretch = cgm_data.loc[(cgm_data['date_time'] >= start) & (cgm_data['date_time'] <= end)][
                'Sensor Glucose (mg/dL)'].values.tolist()

            if len(cgm_data_stretch) < reqLength:
                continue;
            meal_no_meal_data.append(cgm_data_stretch[:reqLength])

    return pd.DataFrame(meal_data), ground_truth, int(no_of_bins)



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


def absoluteValueMean(param):
    meanValue = 0
    for p in range(0, len(param) - 1):
        meanValue = meanValue + np.abs(param[(p + 1)] - param[p])
    return meanValue / len(param)

def calculate_kmeans_clusters(labels, data):

    no_of_clusters = len(set(labels)) if -1 not in labels else len(set(labels)) - 1
    clusters = []
    for i in range(no_of_clusters):
        clusters.append(pd.DataFrame())

    for i in range(len(labels)):
        if labels[i] != -1:
            clusters[labels[i]] = clusters[labels[i]].append(data.iloc[i])
            clusters[labels[i]].reset_index().drop(columns='index')

    return clusters

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
        dict['ArgMaxMinFeature'] = np.argmax(value) - np.argmin(value)

        dict['ZeroCrossingMax1'] = calculate_max_zero_crossings(value , 0)[0]
        dict['ZeroCrossingMaxIndex1'] = calculate_max_zero_crossings(value, 0)[1]
        dict['ZeroCrossingMax2'] = calculate_max_zero_crossings(value, 1)[0]
        dict['ZeroCrossingMaxIndex2'] = calculate_max_zero_crossings(value, 1)[1]
        dict['ZeroCrossingMax3'] = calculate_max_zero_crossings(value, 2)[0]
        dict['ZeroCrossingMaxIndex3'] = calculate_max_zero_crossings(value, 2)[1]

        dict['MaxFFTAmp1'] = calculate_fast_fourier(value, 1)
        dict['MaxFFTAmp2'] = calculate_fast_fourier(value, 2)
        dict['MaxFFTAmp3'] = calculate_fast_fourier(value, 3)

        dict['AbsoluteValueMean1'] = absoluteValueMean(value[:13])
        dict['AbsoluteValueMean2'] = absoluteValueMean(value[13:])
        # Append all the features into features data frame
        features = features.append(dict, ignore_index=True)

    features = pd.DataFrame(features, dtype=float)

    # Principle Component Analysis is done on the features extracted above
    std = StandardScaler().fit_transform(features)
    features = pd.DataFrame(PCA(n_components=5).fit_transform(std))

    return features


def required_clusters_db_scan(dbscan, meal_data, required_clusters):
    '''
    The function increases the number of clusters by splitting clusters that high high error
    This is done until we reach our required number of clusters
    '''

    clusters = calculate_kmeans_clusters(dbscan.labels_, meal_data)
    while (len(clusters) < required_clusters):
        cluster_centers = [cluster.mean(axis=0) for cluster in clusters]
        clusters_sse = [calculate_squared_sum_error(clusters[i].iloc[:, :-1].to_numpy(), cluster_centers[i][:-1].to_numpy()) for i in
                        range(len(clusters))]
        index = argmax(clusters_sse)
        to_split_cluster = clusters[index]
        kMeans = KMeans(n_clusters=2, random_state=0).fit(to_split_cluster.iloc[:, :-1])
        new_clusters = calculate_kmeans_clusters(kMeans.labels_, to_split_cluster)
        del clusters[index]
        for c in new_clusters:  clusters.append(c)

    return clusters



def calculate_squared_sum_error(cluster_data, cluster_center):
    norms = [np.linalg.norm(data - cluster_center) for data in cluster_data]
    return np.sum(np.square(norms))


def entropy_calculation(cluster_data):
    '''
     The function calculates entropy of a cluster
    '''
    total_data_points = np.sum(cluster_data)
    return sum([-1 * (d / total_data_points) * log(d / total_data_points, 2) if d != 0 else 0 for d in cluster_data])


def purity_calculation(cluster_data):
    '''
     The function calcluates purity of a cluster
    '''
    total_data_points = np.sum(cluster_data)
    return max([d / total_data_points for d in cluster_data])


def calculate_required_metrics(clusters, no_of_bins):

    '''
    The function takes in the cluster and number of bins and computes metrics like Squared sum error,
    entropy and purity after performing necessary calculations
    :return: metrics
    '''
    cluster_centers = [cluster.mean(axis=0) for cluster in clusters]

    clusters_sse = [calculate_squared_sum_error(clusters[i].iloc[:, :-1].to_numpy(), cluster_centers[i][:-1].to_numpy()) for i in
                    range(len(clusters))]
    square_sum_error_total = np.sum(clusters_sse)

    cluster_bin_matrix = np.empty([no_of_bins, no_of_bins])
    cluster_bin_matrix.fill(0)

    for idx, cluster in enumerate(clusters):
        for i in range(cluster.shape[0]):
            cluster_bin_matrix[idx][int(cluster.iloc[i, -1]) - 1] += 1

    entropy_cluster = [entropy_calculation(cluster) for cluster in cluster_bin_matrix]
    entropy_total = sum([sum(cluster_bin_matrix[i]) * e for i, e in enumerate(entropy_cluster)]) / np.sum(
        cluster_bin_matrix)

    purity_cluster = [purity_calculation(cluster) for cluster in cluster_bin_matrix]
    purity_total = sum([sum(cluster_bin_matrix[i]) * p for i, p in enumerate(purity_cluster)]) / np.sum(
        cluster_bin_matrix)

    return square_sum_error_total, entropy_total, purity_total



# Read data from the file CGMData.csv

cgm_data = pd.read_csv('CGMData.csv', low_memory=False, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])

# Read data from InsulinData.csv
insulin_data = pd.read_csv('InsulinData.csv', low_memory=False, usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])

# Create a date_time field in cgm data by combining the data and time
cgm_data['date_time'] = pd.to_datetime(
    cgm_data['Date'].astype(str) + " " + cgm_data['Time'].astype(str))

# Create a date_time field in insulin data by combining the data and time

insulin_data['date_time'] = pd.to_datetime(
    insulin_data['Date'].astype(str) + " " + insulin_data['Time'].astype(str))

# Capture meal data and get the ground truth of each data item
# meal_data, ground_truth, bins = extract_ground_truth_from_meal_data(cgm_data, insulin_data)
meal_data, ground_truth, bins = capture_data_meal_no_meal(cgm_data, insulin_data, True)

# Capture  the features for the meal data
meal_features = extractFeaturesFromMealNoMealData(meal_data)

meal_features['ground_truth'] = ground_truth

# Perform K-means Clustering on the meal data features
kMeans = KMeans(n_clusters=bins, random_state=0).fit(meal_features.iloc[:, :-1])

# Dividing sample into clusters
kMeans_clusters = calculate_kmeans_clusters(kMeans.labels_, meal_features)

#  K-means Clustering: Calculate metrics Squared Sum error, Entropy and Purity
sse_kmeans, entropy_kmeans, purity_kmeans = calculate_required_metrics(kMeans_clusters, bins)

# Perform DBScan  on the meal data features
dbscan = DBSCAN(eps=1.4, min_samples=10).fit(meal_features.iloc[:, :-1])

# Calculating required extra clusters as dbscan provides less than number of clusters
clusters_dbscan = required_clusters_db_scan(dbscan, meal_features, bins)

#  DBScan: Calculate metrics Squared Sum error, Entropy and Purity

squared_sum_error_dbscan, dbscan_entropy, dbscan_purity = calculate_required_metrics(clusters_dbscan, bins)

# Create a results data frame to store final results in csv file

results_df = pd.DataFrame()

# Store all the required fields in results data frame
results_df['sse_kmeans'], results_df['sse_dbscan'] = [sse_kmeans], [squared_sum_error_dbscan]
results_df['entropy_kmeans'], results_df['entropy_dbscan'] = [entropy_kmeans], [dbscan_entropy]
results_df['purity_kmeans'], results_df['purity_dbscan'] = [purity_kmeans], [dbscan_purity]

# Add the data frame to Results.csv
results_df.to_csv('Results.csv', header=False, index=False)