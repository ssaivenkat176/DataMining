# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd

''' Function to calculate the metrics for a subsegment (daytime,overnight and whole day) of CGM data'''

def calculatePercentageMetrics(df):
    result = []
    # Number of days present in the dataframe to calculate mean
    num_of_days = (df.groupby([df['Date']]).count()).count()[0];
    # Calculating mean of the count of cgm data where Sensor Glucose (mg/dL) > 180
    df_hyperglycemia = (((df[(df['Sensor Glucose (mg/dL)'] > 180)].groupby([df['Date']]).size() / 288) * 100).sum(axis=0)) / num_of_days
    result.append(df_hyperglycemia)
    # Calculating mean of the count of cgm data where Sensor Glucose (mg/dL) > 250
    df_hyperglycemia_critical = (((df[df['Sensor Glucose (mg/dL)'] > 250].groupby([df['Date']]).size() / 288) * 100).sum(axis=0)) / num_of_days
    result.append(df_hyperglycemia_critical)

    # Calculating mean of the count of cgm data where Sensor Glucose (mg/dL) > = 70 and Sensor Glucose (mg/dL) <= 180
    df_range = (((df[(df['Sensor Glucose (mg/dL)'] >= 70) & (df['Sensor Glucose (mg/dL)'] <= 180)].groupby([df['Date']]).size() / 288) * 100).sum(axis=0)) / num_of_days
    result.append(df_range)

    # Calculating mean of the count of cgm data where Sensor Glucose (mg/dL) > = 70 and Sensor Glucose (mg/dL) <= 150
    df_range_second = (((df[(df['Sensor Glucose (mg/dL)'] >= 70) & (df['Sensor Glucose (mg/dL)'] <= 150)].groupby([df['Date']]).size() / 288) * 100).sum(axis=0)) / num_of_days
    result.append(df_range_second)
    # Calculating mean of the count of cgm data where Sensor Glucose (mg/dL) < 70
    df_range_second_level1 = (((df[(df['Sensor Glucose (mg/dL)'] < 70)].groupby([df['Date']]).size() / 288) * 100).sum(axis=0))/ num_of_days
    result.append(df_range_second_level1)

    # Calculating mean of the count of cgm data where Sensor Glucose (mg/dL) < 54
    df_range_second_level2 = (((df[df['Sensor Glucose (mg/dL)'] < 54].groupby([df['Date']]).size() / 288) * 100).sum(axis=0)) / num_of_days
    result.append(df_range_second_level2)
    return result



if __name__ == '__main__':

    # List to store the columns of InsulinData.csv so we can drop remaining columns
    insulin_fields = ['Date', 'Time','Alarm']
    # List to store the columns of CGMData.csv so we can drop remaining columns
    cgm_fields = ['Date', 'Time','Sensor Glucose (mg/dL)']

    # Read the CGMData.csv file as a data frame
    cgm_df = pd.read_csv("CGMData.csv", encoding="ISO-8859-1",low_memory=False,usecols=cgm_fields)
    # Handle missing values using linear interpolation
    cgm_df['Sensor Glucose (mg/dL)'] = cgm_df['Sensor Glucose (mg/dL)'].interpolate(method='linear', limit_direction='both')

    # Read the InsulinData.csv file as a data frame
    insulin_file_df = pd.read_csv("InsulinData.csv", encoding="ISO-8859-1",low_memory=False,usecols=insulin_fields)

    # Filter out all the data which does not contain Alarm column value as 'AUTO MODE ACTIVE PLGM OFF'
    insulin_file_df = insulin_file_df[insulin_file_df['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF']
    # Combine the Date and Time from the cgm data frame and store it in a new column called DateTime
    cgm_df['DateTime'] = pd.to_datetime(cgm_df['Date'].astype(str) + ' ' + cgm_df['Time'].astype(str))

    # Sort the cgm data in ascending order based on DateTime field
    cgm_df = cgm_df.sort_values(["DateTime"], ascending=True)

    # Combine the Date and Time from the insulin data frame and store it in a new column called DateTime

    insulin_file_df['DateTime'] = pd.to_datetime(insulin_file_df['Date'].astype(str) + ' ' + insulin_file_df['Time'].astype(str),
                                                 format="%m/%d/%Y %H:%M:%S")

    # Sort the insulin data in ascending order based on DateTime field 'AUTO MODE ACTIVE PLGM OFF' at first
    insulin_file_df = insulin_file_df.sort_values(['DateTime'], ascending=True)

    # Find the date and time from insulin data where you encounter
    my_date_time = insulin_file_df['DateTime'].iloc[0]

    # Segregate the cgm data into Manual and Auto modes based on date and time foudn from Insulin Data
    manual_mode_df = cgm_df.loc[cgm_df['DateTime'] < my_date_time]
    auto_mode_df = cgm_df.loc[cgm_df['DateTime'] >= my_date_time]

    # Set the index of manual mode data frame and auto mode frame
    manual_mode_df.set_index(['DateTime'],inplace=True)
    auto_mode_df.set_index(['DateTime'],inplace=True)

    # Create a data frame which stores cdm data during the data from 6 am to midnight - Auto Mode
    daytime_df  = auto_mode_df.between_time('06:00:00', '23:59:59')

    # Create a data frame which stores cdm data during the night from 12 am to 6 am - Auto Mode
    night_time_df  = auto_mode_df.between_time('00:00:00', '05:59:59')

    # Create a data frame which stores cdm data during the whole day from 12 am to 12 am - Auto Mode
    full_day_df = auto_mode_df.between_time('0:00:00','23:59:59')

    # call function to calculate the metrics during the daytime - Auto Mode
    auto_res_day_time = calculatePercentageMetrics(daytime_df)
    # call function to calculate the metrics during the night time - Auto Mode
    auto_res_night_time = calculatePercentageMetrics(night_time_df)

    # call function to calculate the metrics during the whole day - Auto Mode
    auto_res_full_day_time = calculatePercentageMetrics(full_day_df)

    # Create a data frame which stores cdm data during the data from 6 am to midnight - Manual Mode
    daytime_man_df  = manual_mode_df.between_time('06:00:00', '23:59:59')

    # Create a data frame which stores cdm data during the night from 12 am to 6 am - Manual Mode
    night_man_time_df  = manual_mode_df.between_time('00:00:00', '05:59:59')

    # Create a data frame which stores cdm data during the whole day from 12 am to 12 am - Manual Mode
    full_man_day_df = manual_mode_df.between_time('0:00:00','23:59:59')

    # call function to calculate the metrics during the daytime,night mode and full day  - Manual Mode
    man_res_day_time = calculatePercentageMetrics(daytime_man_df)
    man_res_night_time = calculatePercentageMetrics(night_man_time_df)
    man_res_full_time = calculatePercentageMetrics(full_man_day_df)
    # result_manual and result_auto is a variable to store the 18 metrics in Manual mode and Auto Mode respectively.
    result_manual, result_auto = [],[]

    result_manual.extend(man_res_night_time)
    result_manual.extend(man_res_day_time)
    result_manual.extend(man_res_full_time)

    result_auto.extend(auto_res_night_time)
    result_auto.extend(auto_res_day_time)
    result_auto.extend(auto_res_full_day_time)

    result_df = pd.DataFrame()
    # Appending 1.1 as the 19th value in each of the two rows.
    result_manual.append(1.1)
    result_auto.append(1.1)
    # Creating the result data frame
    result_df = result_df.append([result_manual,result_auto], ignore_index=True)
    result_df.fillna(0)
    result_df.to_csv(r'Results.csv',header=False,index=False)

