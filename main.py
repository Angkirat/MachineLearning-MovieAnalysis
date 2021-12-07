import pandas as pd
from pandas.core.reshape.merge import merge
from scipy.sparse import data
import DataCleaning as dc
import time
import modeling as model
from scrape import get_stats

dataCleaning = {
    'SingleValue_onehotEncoding': ['View Rating','Runtime']
    ,'MultiValue_onehotEncoding': ['Genre', 'Tags', 'Languages', 'Country Availability']
    ,'booleanColumns': ['Series or Movie']
    ,'convertBoolean': ['Awards Received','Awards Nominated For']
    ,'numericCleaning': ['viewCount','likeCount','dislikeCount','favoriteCount','commentCount', 'IMDb Votes']
    ,'columnsToRemove': ['Title','Director','Writer','Actors','Production House','Netflix Link','IMDb Link',
                         'Summary','Image','Poster','trailer_link','Trailer Site','video_id','stats','kind','etag',
                         'items','pageInfo.totalResults','pageInfo.resultsPerPage','Boxoffice']
    ,'targetColumn': ['Hidden Gem Score','IMDb Score','Rotten Tomatoes Score','Metacritic Score', 'IMDb Votes']
    ,'dateColumn': ['Release Date','Netflix Release Date']
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

def convert_boolean(actualDF: pd.DataFrame):
    for col in dataCleaning['booleanColumns']:
        actualDF[col] = dc.binary_column(actualDF[col])
    for col in dataCleaning['convertBoolean']:
        actualDF[col] = actualDF[col].isna().astype(int)
    return actualDF

def numeric_data_cleaning(acutalDF:pd.DataFrame):
    acutalDF = dc.numeric_column_cleaning(acutalDF)
    for col in dataCleaning['numericCleaning']:
        acutalDF[col] = dc.standard_scaler(acutalDF[col])
    return acutalDF

def data_cleaning_operation(inputDF: pd.DataFrame):
    inputDF = single_onehot_encoding(inputDF)
    inputDF = multi_onehot_encoding(inputDF)
    inputDF = convert_boolean(inputDF)
    inputDF = numeric_data_cleaning(inputDF)
    inputDF.drop(dataCleaning['SingleValue_onehotEncoding'], axis=1, inplace=True)
    inputDF.drop(dataCleaning['MultiValue_onehotEncoding'], axis=1, inplace=True)
    return inputDF

def clean_data(DF:pd.DataFrame, yt_stats:pd.DataFrame):
    start_time = time.time()
    DF = DF.rename(columns={'TMDb Trailer': 'trailer_link'}).drop_duplicates(subset=['trailer_link'])
    yt_stats = yt_stats.drop_duplicates(subset=['trailer_link'])
    complete_DF = pd.merge(DF, yt_stats, how="inner", on="trailer_link")
    complete_DF.drop(dataCleaning['columnsToRemove'], axis=1, inplace=True)
    complete_DF = complete_DF.drop(complete_DF[complete_DF['viewCount'].isna()].index)
    cleaned_DF = data_cleaning_operation(complete_DF)
    cleaned_DF['FinalScore'] = cleaned_DF[dataCleaning['targetColumn']].mean(axis=1)
    cleaned_DF['target_column'] = cleaned_DF['FinalScore'] < cleaned_DF['FinalScore'].mean()
    cleaned_DF.drop('FinalScore', axis=1, inplace=True)
    cleaned_DF.drop(dataCleaning['targetColumn'], axis=1, inplace=True)
    cleaned_DF.drop(dataCleaning['dateColumn'], axis=1, inplace=True)
    print(cleaned_DF.shape)
    cleaned_DF.dropna(how="any", inplace=True)
    print(cleaned_DF.shape)
    NumericColumns = dataCleaning['numericCleaning'] + dataCleaning['targetColumn']
    cleaningColumns = [col for col in cleaned_DF.columns if (col not in NumericColumns) or (col != 'target_column')]
    print(f'columns to clean are {len(cleaningColumns)}')
    row_count = cleaned_DF.shape[0]
    for col in cleaningColumns:
        col_sum = cleaned_DF[col].sum()
        if col_sum == 0:
            cleaned_DF.drop(col, axis=1, inplace=True)
        elif col_sum == row_count:
            cleaned_DF.drop(col, axis=1, inplace=True)
    print(cleaned_DF.shape)
    cleaned_DF.columns = [col.replace(' ', '_') for col in cleaned_DF.columns]
    return cleaned_DF


    

if __name__ == '__main__':
    DF = pd.read_excel('FlixGem.com Dataset - Latest Netflix data with thousands of attributes.xlsx',sheet_name='FlixGem.com dataset')
    yt_status = get_stats(DF['TMDb Trailer'])
    clean_data = clean_data(DF, yt_status)
    model.main(clean_data, 'target_column')
    pass

