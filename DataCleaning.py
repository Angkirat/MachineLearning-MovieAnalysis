import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import itertools


def one_hot_single(column: pd.Series) -> pd.DataFrame:
    oneHot = OneHotEncoder(handle_unknown="ignore")
    col = column.values.reshape((-1, 1))
    encodedCol = oneHot.fit_transform(col)
    return pd.DataFrame(encodedCol.toarray(), columns=oneHot.categories_[0])


def one_hot_multi(columns: pd.Series, colsep: str = ',') -> pd.DataFrame:
    valueCol = columns.apply(lambda x: [val.strip()
                             for val in x.split(colsep)])
    distinctVal = list(set(itertools.chain(*list(valueCol))))
    outputdf = pd.DataFrame()
    for val in distinctVal:
        outputdf[val] = valueCol.apply(lambda x: val in x)
    return outputdf


def binary_column(columns: pd.Series) -> pd.Series:
    df = pd.get_dummies(columns)
    return df.iloc[:, 1]

def standard_scaler(column: pd.Series) -> pd.DataFrame:
    scaler = StandardScaler()
    return (pd.DataFrame(scaler.fit_transform(column.to_frame()), columns=[column.name])[column.name])

def numeric_column_cleaning(df: pd.DataFrame) -> pd.DataFrame:
  df['IMDb Score']=df['IMDb Score']/10
  df['Rotten Tomatoes Score']=df['Rotten Tomatoes Score']/100
  df['Rotten Tomatoes Score'].fillna(df['IMDb Score'], inplace=True)
  df['IMDb Score'].fillna(df['Rotten Tomatoes Score'], inplace=True)
  df['Metacritic Score']=df['Metacritic Score']/100
  df['Metacritic Score'].fillna(df['IMDb Score'], inplace=True)
  return df
