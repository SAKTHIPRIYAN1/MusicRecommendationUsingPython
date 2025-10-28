import pandas as pd
df = pd.read_csv("dataset/music.csv")
print(df.columns)
print(df[['Title', 'Genres']].head(5))
