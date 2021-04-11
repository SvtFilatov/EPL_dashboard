import pandas as pd
from understatapi import UnderstatClient
import warnings
warnings.filterwarnings('ignore')
understat = UnderstatClient()

def get_first_team(s):                       #Создаем список участников в каждом сезоне
    return s.split(',')[0]

seasons = ['2014','2015','2016','2017','2018','2019','2020']
teams = [list(understat.league(league="EPL").get_player_data(season=i)['team_title'].apply(get_first_team).unique()) for i in seasons]

all_matches = []                                   #Создаем дата-сет из всех матчей
for i,season in enumerate(seasons):
    for team in teams[i]:
        data = understat.team(team=team).get_match_data(season=season)
        data['season'] = season
        all_matches.append(data)
big_data_set = pd.concat(all_matches)
big_data_set = big_data_set.sort_values('datetime').reset_index()
big_data_set['team_h'] = big_data_set['h'].apply(lambda x: x['title'])
big_data_set['team_a'] = big_data_set['a'].apply(lambda x: x['title'])
big_data_set['goals_h'] = big_data_set[big_data_set['isResult']== True]['goals'].apply(lambda x: float(x['h']))
big_data_set['goals_a'] = big_data_set[big_data_set['isResult']== True]['goals'].apply(lambda x: float(x['a']))
big_data_set['xG_h'] = big_data_set[big_data_set['isResult']== True]['xG'].apply(lambda x: float(x['h']))
big_data_set['xG_a'] = big_data_set[big_data_set['isResult']== True]['xG'].apply(lambda x: float(x['a']))
big_data_set['prob_h'] = big_data_set[big_data_set['isResult']== True]['forecast'].apply(lambda x: float(x['w']))
big_data_set['prob_d'] = big_data_set[big_data_set['isResult']== True]['forecast'].apply(lambda x: float(x['d']))
big_data_set['prob_a'] = big_data_set[big_data_set['isResult']== True]['forecast'].apply(lambda x: float(x['l']))
big_data_set = big_data_set[big_data_set['side'] == 'h'].drop(['h','a','goals', 'xG', 'side', 'forecast'], axis = 1)

other_matches = big_data_set[big_data_set['isResult']== False]
big_data_set = big_data_set[big_data_set['isResult']== True]

big_data_set.to_csv('big_matches_set.csv')
other_matches.to_csv('next_games.csv')