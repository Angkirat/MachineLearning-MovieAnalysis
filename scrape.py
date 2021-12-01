import pandas as pd
import json
import requests

# Youtube Data API Key
API_KEY = "AIzaSyBi0lQPCDMotyUg_8KIMP9-_DwAp_o2BtI"

def get_video_data(video_id, api_key): 
	if video_id == None:
		return ""
	url = f"https://www.googleapis.com/youtube/v3/videos?part=statistics&id={video_id}&key={api_key}"
	json_url = requests.get(url)
	data = json.loads(json_url.text)
	try:
		data = data['items'][0]['statistics']
	except (KeyError):
		print(f'ERR: YouTube Scrape: {video_id} - {data["error"]["message"]}')
		data = dict()
	except (IndexError):
		print(f'ERR: YouTube Scrape: {video_id}')
	return data

def get_stats(trailer_links: list):
	df = pd.DataFrame(trailer_links, columns = ['trailer_link'])
	df['video_id'] = df.iloc[:, 0].str.split("=", expand=True)[1]
	df['stats'] = df.apply(lambda row: get_video_data(row['video_id'], API_KEY), axis=1)
	return pd.concat([df, pd.json_normalize(df['stats'])], axis=1)