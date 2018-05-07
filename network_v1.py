import pandas as pd
import tensorflow as tf
import numpy as np

df = pd.read_csv("data/Ch2_001/final_example.csv");

rows, cols = df.shape

"""
 Assign label to each image
 1 = right 
 0 = straight 
 -1 = left
"""
class_column = pd.Series(np.zeros(rows, dtype=int));
for index in range(rows):
    angle = df.loc[index, 'steering_angle']
    if angle > 0.1:
        class_column[index] = 1
    elif angle < -0.1:
        class_column[index] = -1
df['class'] = class_column


print(df.head())
print(df['class'].value_counts())
