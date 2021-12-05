import pandas as pd
from pandas.core.reshape.merge import merge
import DataCleaning as dc
# from scrape import get_stats

dataCleaning = {
    'SingleValue_onehotEncoding': ['View Rating','Runtime']
    ,'MultiValue_onehotEncoding': ['Genre', 'Tags', 'Languages', 'Country Availability']
}

def single_onehot_encoding(acutalDF: pd.DataFrame):
    df = pd.DataFrame()
    for col in dataCleaning['SingleValue_onehotEncoding']:
        cleanDF = dc.one_hot_single(acutalDF[col])
        cleanDF.columns = [f'{col}_{st}' for st in cleanDF.columns]
        df = pd.concat([df, cleanDF], axis=1)
    return pd.concat([acutalDF, df], axis=1)

def multi_onehot_encoding(acutalDF: pd.DataFrame):
    df = pd.DataFrame()
    for col in dataCleaning['MultiValue_onehotEncoding']:
        cleanDF = dc.one_hot_single(acutalDF[col])
        cleanDF.columns = [f'{col}_{st}' for st in cleanDF.columns]
        df = pd.concat([df, cleanDF], axis=1)
    return pd.concat([acutalDF, df], axis=1)


def data_cleaning_operation(inputDF: pd.DataFrame):
    inputDF = single_onehot_encoding(inputDF)
    inputDF = multi_onehot_encoding(inputDF)
    inputDF['isSeries'] = dc.binary_column(inputDF['Series or movie'])
    inputDF.drop(dataCleaning['SingleValue_onehotEncoding'], axis=1, inplace=True)
    inputDF.drop(dataCleaning['MultiValue_onehotEncoding'], axis=1, inplace=True)
    return inputDF

def main():
    DF = pd.read_excel('FlixGem.com Dataset - Latest Netflix data with thousands of attributes.xlsx',sheet_name='FlixGem.com dataset')
    yt_stats = pd.read_csv('yt_stats.csv').iloc[:, 1:]
    complete_DF = DF.merge(yt_stats, left_on='TMDb Trailer', right_on='trailer_link')
    
    print(DF.shape)
    print(yt_stats.shape)
    print(complete_DF.shape)




if __name__ == '__main__':
    main()
    pass

