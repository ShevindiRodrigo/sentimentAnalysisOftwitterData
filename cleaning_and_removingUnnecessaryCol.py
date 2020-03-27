import pandas as pd
import re
df = pd.read_csv('/home/user/Documents/research dataset/HutchSriLanka_tweets.csv', encoding = 'unicode_escape')
df.head(5)
df=df.drop_duplicates(['tweet'])

print(df['id'].isna().sum())
print(df['created_at'].isna().sum())
print(df['tweet'].isna().sum())




for i in range(len(df)):
    txt = df.loc[i]["tweet"]
    txt=re.sub(r'@[A-Z0-9a-z_:]+','',txt)#replace username-tags
    txt=re.sub(r'^[RT]+','',txt)#replace RT-tags
    txt = re.sub('https?://[A-Za-z0-9./]+','',txt)#replace URLs
    txt=re.sub("[^a-zA-Z]", " ",txt)#replace hashtags
    df.at[i,"tweet"]=txt
    df.to_csv('/home/user/Documents/research dataset/cleanedhutch_tweets.csv', sep='\t', encoding='utf-8')
