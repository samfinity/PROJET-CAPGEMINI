import pandas as pd
import numpy as np

df = pd.read_csv('data_music_project.csv.gz', compression='gzip')
print(df.head())