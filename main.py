import pandas as pd
from scrape import get_stats

DF = pd.read_excel('FlixGem.com Dataset - Latest Netflix data with thousands of attributes.xlsx', sheet_name='FlixGem.com dataset')

def main():
	yt_stats = get_stats(DF["TMDb Trailer"][DF["Trailer Site"] == "YouTube"].tolist())
	print(yt_stats)

if __name__ == '__main__':
    main()
    pass