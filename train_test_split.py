import pandas as pd
import numpy as np

df = pd.read_csv('/home/user/Documents/research dataset/sentiScoreDialog_tweets.csv')
#df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.7

train = df[msk]
test = df[~msk]
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
