import requests        #FROM: https://www.kaggle.com/abrarhossainhimself/understat-data-for-teams-players-2014-present
import json
import pandas as pd
import os

def scrape_understat(payload):
    url = 'https://understat.com/main/getPlayersStats/'
    headers = {'content-type':'application/json; charset=utf-8',
    'Host': 'understat.com',
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:73.0) Gecko/20100101 Firefox/73.0',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Encoding': 'gzip, deflate, br',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'X-Requested-With': 'XMLHttpRequest',
    'Content-Length': '39',
    'Origin': 'https: // understat.com',
    'Connection': 'keep - alive',
    'Referer': 'https: // understat.com / league / EPL'
    }
    response = requests.post(url, data=payload, headers = headers, verify=True)
    response_json = response.json()
    inner_wrapper = response_json['response']
    json_player_data = inner_wrapper['players']
    return json_player_data

def clean_df(player_df, weeks):
    player_df  = player_df.rename(columns={'goals':'goals_'+weeks,'xG':'xG_'+weeks,'assists':'assists_'+weeks, 'xA':'xA_'+weeks, 'shots':'shots_'+weeks, 'key_passes':
        'key_passes_'+weeks,'npg':'npg_'+weeks,'npxG':'npxG_'+weeks})
    return(player_df)

def gw_data(no_of_gw, season = '2020'):
    json_player_data = scrape_understat({'league':'EPL', 'season':season, 'n_last_matches': no_of_gw})
    gw_table = pd.DataFrame(json_player_data)
    gw_df = clean_df(gw_table,str(no_of_gw)+'wks')
    gw_df['position'] = gw_df['position'].str.slice(0,1)
    position_map = {'D':'DEF', 'F':'FWD', 'M':'MID', 'G':'GK', 'S':'FWD'}
    gw_df = gw_df.replace({'position': position_map})
    gw_df['team_title'] = gw_df['team_title'].apply(lambda x: x.split(',')[-1])
    return gw_df

df = gw_data(no_of_gw = 5)
df['xGChain'] = df['xGChain'].apply(float)

df.to_csv('players_ds.csv')        #END FROM