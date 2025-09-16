import pandas as pd
import numpy as np

df = pd.read_csv('all_seasons.csv')

sample_df = df.sample(n=100, random_state=42)

print(sample_df.head())
