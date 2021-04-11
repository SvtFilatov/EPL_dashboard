import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from bokeh.models import ColumnDataSource
from sklearn.linear_model import LinearRegression
import random

st.title('English Premier League interactive dashboard')

def to_time(s):
    return datetime.strptime(s[2:], '%y-%m-%d %H:%M:%S')
begining = to_time('2014-08-01 00:00:00')

st.write('This is a powerful tool for analyzing english football. Our goal was to give a ')
st.write('numeric description for attacking and defensive power of every team in every moment ')
st.write("of time. We've built a rating system based on [understat.com](https://understat.com) advanced measures (xG, ")
st.write("xGChain, etc.) and [Glicko-2 model](https://en.wikipedia.org/wiki/Glicko_rating_system), which is used for describing skills of players in")
st.write("chess, for instance. You can find the realization of this algorithm in the corresponding")
st.write("repository on GitHub in the file 'create_indexes.py'. As you can see further, these")
st.write("ratings can describe both team's strength and current form. ")
st.write('_')
st.write('This graph represents current strength of defensive and attacking lines in every EPL')
st.write('team. We also give a list of 3 best players in every team in terms of attacking impact.')
st.write('Attacking impact is measured as a percent of expected goals created during possessions in')
st.write('which player participated from all expected goals created by the team in past 5 games.')

c_d = {'Manchester United': '#DA291C' , 'West Ham': '#7A263A', 'Queens Park Rangers': '#87CEFA',
       'West Bromwich Albion': '#122F67', 'Stoke':'#E03A3E', 'Leicester':'#003090', 'Arsenal':'#EF0107',
       'Liverpool': '#C8102E', 'Newcastle United':'#241F20', 'Burnley': '#6C1D45', 'Aston Villa' : '#95BFE5',
       'Chelsea':'#034694', 'Swansea':'#121212', 'Southampton':'#D71920', 'Crystal Palace':'#1B458F',
       'Everton':'#003399', 'Hull':'#FF8C00', 'Tottenham':'#132257', 'Sunderland':'#FF0000',
       'Manchester City':'#6CABDD','Bournemouth':'#B50E12', 'Norwich':'#FFF200', 'Watford':'#FBEE23',
       'Middlesbrough':'#DC143C', 'Brighton':'#0057B8', 'Huddersfield':'#0E63AD', 'Fulham':'#CC0000',
       'Wolverhampton Wanderers':'#FDB913', 'Cardiff':'#0070B5', 'Sheffield United':'#EE2737', 'Leeds':'#FFCD00'}
color_dict = {}
for i in sorted(c_d.keys()):
    color_dict[i] = c_d[i]

big_data_set = pd.read_csv('big_matches_set.csv')
other_matches = pd.read_csv('next_games.csv')
curr_table = pd.read_csv('current ratings.csv').set_index('Unnamed: 0')

def get_data_for_team(name):
    df = pd.read_csv('teams time series/'+name+' stats.csv')
    seas = df['season'].unique()
    ans =[]
    for i in seas:
        res = df[df['season'] == i]
        res['date'] = res['date'].apply(to_time)
        ans.append(res[['attacking rate','defending rate','date', 'season', 'match_id', 'result']])
    return ans

TOOLTIPS_m=[
    ("Date", "@date"),
    ("Rate", "@rating"),
    ("Last game", "@result"),
]

values = ['$1.13bn', '$1.11bn', '$857.78m', '$604.01m',
          '$508.75m', '$748.88m', '$789.75m', '$275.55m',
          '$453.31m', '$143.28m', '$223.63m', '$236.12m',
          '$307.73m', '$273.46m', '$277.97m', '$133.38m',
          '$417.67m', '$523.49m', '$168.08m', '$245.41m']

curr_table['total value'] = values
curr_table['1st player'] = 0
curr_table['2nd player'] = 0
curr_table['3rd player'] = 0

df = pd.read_csv(r'players_ds.csv')

for team in curr_table.index:
    res = df[df['team_title'] == team].sort_values('xGChain', ascending = False)
    total_xg = df[df['team_title'] == 'Tottenham']['xG_5wks'].apply(float).sum()
    curr_table['1st player'][team] = list(res['player_name'])[0] + ' (' + str(int(round(list(res['xGChain'])[0]/total_xg, 2)*100)) + '%)'
    curr_table['2nd player'][team] = list(res['player_name'])[1] + ' (' + str(int(round(list(res['xGChain'])[1]/total_xg, 2)*100)) + '%)'
    curr_table['3rd player'][team] = list(res['player_name'])[2] + ' (' + str(int(round(list(res['xGChain'])[2]/total_xg, 2)*100)) + '%)'

TOOLTIPS=[
    ("Title", "@title"),
    ("Squad market value", "@value"),
    ('1st player by attacking impact', "@first_player"),
    ('2nd player', "@second_player"),
    ('3rd player', "@third_player")
]

source = ColumnDataSource(data=dict(x=curr_table['attacking rate'], y=curr_table['defending rate'],
                                    title=curr_table.index,
                                    color = [color_dict[i] for i in curr_table.index],
                                    value = curr_table['total value'],
                                    first_player = curr_table['1st player'],
                                    second_player = curr_table['2nd player'],
                                    third_player = curr_table['3rd player']))

corr = figure(x_axis_label='bad offence'+' '*130 + 'good offence',
              y_axis_label= 'bad defence'+' '*110 + 'good defence',
              title="", toolbar_location=None, tooltips=TOOLTIPS, sizing_mode="scale_both")
corr.circle(x='x', y='y', source=source, size=13, line_color=None, color = 'color')
st.bokeh_chart(corr, use_container_width=True)

st.write('Now you can look through the time charts for each team played in EPL in the past 7 years.')
st.write('Each time chart represents attacking (or defensive) power of the team through the time.')
st.write('As well you can see the result of every game played by the team in EPL and its influence')
st.write("on graph's trend. Just pick the team you interested in the most!")
col1, col2 = st.beta_columns(2)

team1 = col1.selectbox('first team', list(color_dict.keys()), index= 15)
team2 = col2.selectbox('second team', list(color_dict.keys()), index = 16)
rate = st.selectbox('rate', ['attacking rate', 'defending rate'])


p = figure(x_axis_label='date', y_axis_label=rate, tooltips=TOOLTIPS_m, x_axis_type="datetime")
for df in get_data_for_team(team1):
    source = ColumnDataSource(data=dict(x = df['date'], y=df[rate],
                                        date = df['date'].apply(str),
                                        rating = df[rate].apply(int),
                                        result = df['result']))
    p.line(x = 'x', y = 'y', source = source, legend_label=team1, line_width=2, line_color='lightblue')
for df in get_data_for_team(team2):
    source = ColumnDataSource(data=dict(x=df['date'], y=df[rate],
                                        date=df['date'].apply(str),
                                        rating=df[rate].apply(int),
                                        result=df['result']))
    p.line(x = 'x', y = 'y', source = source, legend_label=team2, line_width=2, line_color='red')

st.bokeh_chart(p, use_container_width=True)

st.write("So what are the next steps? Well, we've created a powerful tool for building predictive")
st.write('models. For example, we can create a simulation of the EPL 2020/21 ending.')
st.write(' Just push the button and see who is going to make to the top-4!')

simulate = st.button('Simulate!')

all_xg = np.array(list(big_data_set['xG_h'])+list(big_data_set['xG_a']))
scale = np.sort(np.array(random.choices(all_xg, k = 20000)))

just_a_table = pd.DataFrame(index = big_data_set[big_data_set['season'] == 2020]['team_h'].unique(),
                            columns = ['points', 'wins', 'draws',
                                                          'loses', 'matches']).fillna(value = 0)

res = big_data_set[big_data_set['season'] == 2020]
for j in res.index:
    just_a_table['matches'][res['team_h'][j]] += 1
    just_a_table['matches'][res['team_a'][j]] += 1
    if res['result'][j] == 'w':
        just_a_table['points'][res['team_h'][j]] += 3
        just_a_table['wins'][res['team_h'][j]] += 1
        just_a_table['loses'][res['team_a'][j]] += 1
    elif res['result'][j] == 'd':
        just_a_table['points'][res['team_h'][j]] += 1
        just_a_table['points'][res['team_a'][j]] += 1
        just_a_table['draws'][res['team_h'][j]] += 1
        just_a_table['draws'][res['team_a'][j]] += 1
    else:
        just_a_table['points'][res['team_a'][j]] += 3
        just_a_table['wins'][res['team_a'][j]] += 1
        just_a_table['loses'][res['team_h'][j]] += 1

X = big_data_set[['xG_h', 'xG_a']]
Y = big_data_set[['prob_h', 'prob_d','prob_a']]
clf = LinearRegression(normalize=True).fit(X, Y)

def g(x):
    return 1.0 / np.sqrt(1 + 3 * np.power(x / 3.14, 2))


def E(r1, r2, x):
    return 1.0 / (1 + np.exp(-g(x) * (r1 - r2)))

for j in other_matches.index:
    home_team = other_matches['team_h'][j]
    away_team = other_matches['team_a'][j]
    exp_h_scored = E(curr_table['attacking rate'][home_team], curr_table['defending rate'][away_team],
                     curr_table['RD_d'][away_team])
    exp_a_scored = E(curr_table['attacking rate'][away_team], curr_table['defending rate'][home_team],
                     curr_table['RD_d'][home_team])
    other_matches['xG_h'][j] = scale[int(np.around(exp_h_scored*20000))]
    other_matches['xG_a'][j] = scale[int(np.around(exp_a_scored*20000))]

other_matches[['prob_h', 'prob_d', 'prob_a']] = clf.predict(other_matches[['xG_h', 'xG_a']])

just_a_new_table = just_a_table.copy()
for j in other_matches.index:
    home_team = other_matches['team_h'][j]
    away_team = other_matches['team_a'][j]
    just_a_new_table['matches'][home_team] += 1
    just_a_new_table['matches'][away_team] += 1
    flag = random.random()
    if other_matches['prob_h'][j]>flag:
        just_a_new_table['points'][home_team] += 3
        just_a_new_table['wins'][home_team] += 1
        just_a_new_table['loses'][away_team] += 1
    elif (other_matches['prob_h'][j] + other_matches['prob_d'][j] > flag) and (other_matches['prob_h'][j] < flag):
        just_a_new_table['points'][home_team] += 1
        just_a_new_table['points'][away_team] += 1
        just_a_new_table['draws'][home_team] += 1
        just_a_new_table['draws'][away_team] += 1
    else:
        just_a_new_table['points'][away_team] += 3
        just_a_new_table['wins'][away_team] += 1
        just_a_new_table['loses'][home_team] += 1


if simulate == False:
    st.markdown('This is current EPL table. Just push the button and see who is going to make it to the top-4!')
    st.dataframe(just_a_table.sort_values('points', ascending=False), height = 600)
else:
    st.markdown("You've just simulated the season! This is the potential final table.")
    st.dataframe(just_a_new_table.sort_values('points', ascending=False), height = 600)

st.write("You can simulate the table several times as long as you want.")

#python  -m streamlit run main.py

