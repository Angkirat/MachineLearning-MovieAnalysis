import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import itertools

def one_hot_single(column:pd.Series) -> pd.DataFrame:
    oneHot = OneHotEncoder(handle_unknown="ignore")
    col = column.values.reshape((-1,1))
    encodedCol = oneHot.fit_transform(col)
    return pd.DataFrame(encodedCol, columns=oneHot.categories_[0])

def one_hot_multi(columns:pd.Series, colsep:str=',') -> pd.DataFrame:
    valueCol = columns.apply(lambda x: [val.strip() for val in x.split(colsep)])
    distinctVal = list(set(itertools.chain(*list(valueCol))))
    outputdf = pd.DataFrame()
    for val in distinctVal:
        outputdf[val] = valueCol.apply(lambda x: val in x)
    return outputdf

def binary_column(columns: pd.Series) -> pd.Series:
    df = pd.get_dummies(columns)
    return df.iloc[:, 1]
