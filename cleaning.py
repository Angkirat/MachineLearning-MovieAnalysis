import pandas as pd

def removing_na:
  df = pd.read_excel("/content/drive/MyDrive/flixgem.xlsx",sheet_name='FlixGem.com dataset')
  df['IMDb Score']=df['IMDb Score']/10
  df['Rotten Tomatoes Score']=df['Rotten Tomatoes Score']/100
  df['Rotten Tomatoes Score'].fillna(df['IMDb Score'], inplace=True)
  df['IMDb Score'].fillna(df['Rotten Tomatoes Score'], inplace=True)
  df['Metacritic Score']=df['Metacritic Score']/100
  df['Metacritic Score'].fillna(df['IMDb Score'], inplace=True)
  return df
