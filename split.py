import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
SEED=4
df = pd.read_csv('/home/user/Documents/research dataset/cleaneddialog_tweets.csv')
x = df[['tweet']]
y = df[['sentiment']]
x_train,x_val_test,y_train,y_val_test = train_test_split(x,y,test_size=0.1,random_state=SEED)
x_val,x_test,y_val,y_test = train_test_split(x_val_test,y_val_test,test_size=0.5,random_state=SEED)

x_val.to_csv('/home/user/Documents/research dataset/x_val.csv', sep='\t', encoding='utf-8')
x_test.to_csv('/home/user/Documents/research dataset/x_test.csv', sep='\t', encoding='utf-8')
y_val.to_csv('/home/user/Documents/research dataset/y_val.csv', sep='\t', encoding='utf-8')
y_test.to_csv('/home/user/Documents/research dataset/y_test.csv', sep='\t', encoding='utf-8')
