import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import math

FEATURE_VALUES = [
    'precipitation',
    'max_temp',
    'min_temp',
    'avg_wind_speed',
    'year',
    'temp_range',
    'wind_temp_ratio',
    'month',
    'lagged_precipitation',
    'lagged_avg_wind_speed',
]
#one hot encoding for seasons
SEASON_ORDER = ['summer', 'fall', 'winter', 'spring']

TARGET_NAME = 'fire_start_day'


X = [] #FEATURE MATRIX
Y = [] #TARGET VECTOR
column_indices = {} #DICTIONARY STORING COLUMN NAME AND ITS INDEX

#Read in CSV file
with open('CA_Weather_Fire_Dataset_1984-2025.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)

    #Maps columns to their index
    header =  next(reader)[1:13]
    for i, col_name in enumerate(header):
        column_indices[col_name.lower()] = i

    season_index = column_indices['season']
    target_index = column_indices[TARGET_NAME]

    #Iterate through each row of the dataset
    for row in reader:
        data_slice = row[1:13]

        if not data_slice:
            continue
        
        feature_vector = []
        
        #Processes whole row, skips row if any error occurs
        try:
            for name in FEATURE_VALUES:
                idx = column_indices.get(name)

                #One hot encoding for the 'season' features
                if idx == season_index:
                    current_season = data_slice[idx].lower()
                    for category in SEASON_ORDER:
                        is_match = 1 if current_season == category else 0
                        feature_vector.append(is_match)
                    continue
                
                #Process numerical features
                value = data_slice[idx].strip()

                #Check if missing value in columns
                if value == '':
                    raise ValueError("Empty string found")

                feature_vector.append(round(float(value), 5))

            target_value = data_slice[target_index].lower()

            #Append processed row to lists
            X.append(feature_vector)
            Y.append(1 if target_value == 'true' else 0)
        except (ValueError, IndexError) as e:
            pass



    
