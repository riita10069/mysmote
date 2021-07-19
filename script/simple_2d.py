# Simple 2D Data Set
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.datasets import make_classification

df = make_classification(
    n_samples = 10000, n_features = 2, n_informative = 2, n_redundant = 0,
    n_repeated = 0, n_classes = 2, n_clusters_per_class = 2, weights = [0.99, 0.01], 
    flip_y = 0, class_sep = 1.0, hypercube = True, shift = 1.0, 
    scale = 1.0, shuffle = True, random_state = 500)

df_raw = DataFrame(df[0], columns = ['x', 'y'])
df_raw['Class'] = df[1]
df = df_raw

# To Visualize
import seaborn as sns
sns.scatterplot(x='x', y='y', hue='Class', data=df)
