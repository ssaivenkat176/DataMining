from math import ceil

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

    time_of_data, meal_no_meal_data,bolus_data= [],[],[]
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
        bolus_data.append(round(insulin_data['BWZ Estimate (U)'].iloc[i]))
    insulin_bolus_data = []
    if is_meal == True:
        for i in range(len(time_of_data)):

            start = time_of_data[i] - pd.Timedelta(minutes=30)
            end = time_of_data[i] + pd.Timedelta(minutes=120)

            cgm_data_date_time_stretch = \
            cgm_data.loc[(cgm_data['date_time'] >= start) & (cgm_data['date_time'] <= end)][
                'Sensor Glucose (mg/dL)'].values.tolist()

            if len(cgm_data_date_time_stretch) < reqLength:
                continue
            cgm_data_date_time_stretch = np.asarray(cgm_data_date_time_stretch)
            meal_no_meal_data.append(cgm_data_date_time_stretch[:reqLength])
            insulin_bolus_data.append(bolus_data[i])
    else:
        for i in range(len(time_of_data) - 1):
            start = time_of_data[i] + pd.Timedelta(minutes = 120)
            end = time_of_data[i + 1]
            cgm_data_stretch = cgm_data.loc[(cgm_data['date_time'] >= start) & (cgm_data['date_time'] <= end)][
                'Sensor Glucose (mg/dL)'].values.tolist()

            if len(cgm_data_stretch) < reqLength:
                continue

            meal_no_meal_data.append(cgm_data_stretch[:reqLength])

    return meal_no_meal_data,insulin_bolus_data



if __name__ == "__main__":

    # Capture data of patient 1 from CGMData.csv and InsulinData.csv and filter out the columns which are not required
    cgm_data_df = pd.read_csv('CGMData.csv', low_memory=False, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
    insulin_data_df = pd.read_csv('InsulinData.csv', low_memory=False, usecols=['Date', 'Time', 'BWZ Carb Input (grams)'
        ,'BWZ Estimate (U)'])


    # Create a date_time field in cgm data by combining the data and time
    cgm_data_df['date_time'] = pd.to_datetime(
        cgm_data_df['Date'].astype(str) + " " + cgm_data_df['Time'].astype(str))

    # Create a date_time field in insulin data by combining the data and time

    insulin_data_df['date_time'] = pd.to_datetime(
        insulin_data_df['Date'].astype(str) + " " + insulin_data_df['Time'].astype(str))

    # Sort the values of the insulin data
    insulin_data_df = insulin_data_df.sort_values(by='date_time')

    # Filter out the rows where the column BWZ Carb Input (grams) in insulin values in NA or 0
    insulin_data_df = insulin_data_df[(insulin_data_df['BWZ Carb Input (grams)'].notna())
                                      & (insulin_data_df['BWZ Carb Input (grams)'] > 0.0)]

    # Using interpolation to handling missing values in cgm data
    cgm_data_df['Sensor Glucose (mg/dL)'] = cgm_data_df['Sensor Glucose (mg/dL)'].interpolate(method='linear',
                                                                                        limit_direction='both')
    insulin_data_df = insulin_data_df.reset_index().drop(columns='index')

    # Extract meal data
    meal_data,insulin_bolus = capture_data_meal_no_meal(cgm_data_df, insulin_data_df, True)

    cgm_min = float('inf')
    b_max,b_meal = [],[]
    max_cgm_row, time_meal_cgm = [],[]
    # Find the minium cgm over all P*30 values
    for row_index, row in enumerate(meal_data):
        cgm_min = min(cgm_min,min(row))

    for row_index, row in enumerate(meal_data):
        max_cgm = row.max()
        if max_cgm != cgm_min:
           # Find the bin number associated with the maximum cgm in a row
           b_max.append(ceil((max_cgm - cgm_min) / 20))
        else:
            b_max.append(1)
        cgm_meal = row[-6]
        if cgm_meal != cgm_min:
            # Find the bin number associated with cgm associated with meal data
            b_meal.append(ceil((cgm_meal - cgm_min) / 20))
        else:
            b_meal.append(1)
    # Form the itemsets from the data
    itemsets_df = pd.DataFrame({'b_max': b_max, 'b_meal': b_meal, 'insulin_bolus': insulin_bolus})

    # Find the frequent itemsets X,YX
    frequency_itemsets = itemsets_df.groupby(['b_max', 'b_meal', 'insulin_bolus']).size().reset_index(name='frequency')
    max_freq = frequency_itemsets['frequency'].max()

    # Find the most frequent itemsets with maximum value
    most_freq_itemsets = frequency_itemsets.loc[frequency_itemsets['frequency'] == max_freq][['b_max', 'b_meal', 'insulin_bolus']]

    most_freq_itemsets = most_freq_itemsets.apply(lambda x: '{{{0},{1}}} -> {2}'.format(x[0], x[1], x[2]),
                                                              axis=1)
    most_freq_itemsets.to_csv('MostFrequentItemset.csv', header=False, index=False)

    frequent_rules = itemsets_df.groupby(['b_max', 'b_meal']).size().reset_index(name='rule_frequency')
    frequent_rules = pd.merge(frequency_itemsets, frequent_rules, on=['b_max', 'b_meal'])
    # Find the confidence of the rule (X,Y) -> Z
    frequent_rules['confidence'] = frequent_rules['frequency'] / frequent_rules['rule_frequency']
    # Find the value with maximum confidence
    largest_confidence = frequent_rules['confidence'].max()
    # Locate all the rules which have maximum confidence
    largest_confidence_rules = frequent_rules.loc[frequent_rules['confidence'] == largest_confidence][
        ['b_max', 'b_meal', 'insulin_bolus']]
    largest_confidence_rules = largest_confidence_rules.apply(lambda x: '{{{0},{1}}} -> {2}'.format(x[0], x[1], x[2]),
                                                              axis=1)

    largest_confidence_rules.to_csv('ConfidenceRules.csv', header=False, index=False)
    # Filter all the rules whose confidence is less than 0.15
    # These rules are anomalous rules

    anomalous_rules = frequent_rules.loc[frequent_rules['confidence'] < 0.15][['b_max', 'b_meal', 'insulin_bolus']]
    anomalous_rules = anomalous_rules.apply(lambda x: '{{{0},{1}}} -> {2}'.format(x[0], x[1], x[2]), axis=1)
    anomalous_rules.to_csv('AnomalousRules.csv', header=False, index=False)






