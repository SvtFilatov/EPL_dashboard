import pandas as pd
import numpy as np
from understatapi import UnderstatClient
understat = UnderstatClient()
import warnings
warnings.filterwarnings('ignore')
import random
random.seed = 2

big_data_set = pd.read_csv('big_matches_set.csv')
other_matches = pd.read_csv('next_games.csv')

def get_first_team(s):                             #Создаем список участников в каждом сезоне
    return s.split(',')[0]

seasons = ['2014','2015','2016','2017','2018','2019','2020']
teams = [list(understat.league(league="EPL").get_player_data(season=i)['team_title'].apply(get_first_team).unique()) for i in seasons]

all_xg = np.array(list(big_data_set['xG_h'])+list(big_data_set['xG_a']))
scale = np.sort(np.array(random.choices(all_xg, k = 20000)))

def init_rates():
    rating_table = pd.DataFrame(index = ['Manchester City', 'Liverpool', 'Chelsea', 'Arsenal', 'Everton',
                                         'Tottenham', 'Manchester United', 'Southampton', 'Stoke', 'Newcastle United',
                                         'Crystal Palace', 'Swansea', 'West Ham', 'Sunderland', 'Aston Villa', 'Hull',
                                         'West Bromwich Albion', 'Leicester', 'Burnley', 'Queens Park Rangers'])
    rating_table['attacking rate'] = np.linspace(1650.0, 1450.0, 20)
    rating_table['defending rate'] = np.linspace(1600.0, 1400.0, 20)
    rating_table['RD_a'] = [350.0]*20
    rating_table['RD_d'] = [350.0]*20
    rating_table['vol_a'] = [0.06]*20
    rating_table['vol_d'] = [0.06]*20
    return rating_table


def g(x):
    return 1.0 / np.sqrt(1 + 3 * np.power(x / 3.14, 2))


def E(r1, r2, x):
    return 1.0 / (1 + np.exp(-g(x) * (r1 - r2)))


def f(x, delta, rd, v, a, tau):
    return ((np.exp(x) * (np.power(delta, 2) - np.power(rd, 2) - v - np.exp(x)) / (
                2 * np.power(np.power(rd, 2) + v + np.exp(x), 2))) - (x - a) / np.power(tau, 2))


def update_volatility(delta, rd, volatility, v, eps=0.000001, tau=1.2):
    a = np.log(np.power(volatility, 2))
    A = a
    if (delta * delta > rd * rd + v):
        B = np.log(delta * delta - rd * rd - v)
    else:
        k = 1
        while f(a - k * tau, delta, rd, v, a, tau) < 0:
            k = k + 1
        B = a - k * tau
    f_a = f(A, delta, rd, v, a, tau)
    f_b = f(B, delta, rd, v, a, tau)
    while np.absolute(A - B) > eps:
        C = A + (A - B) * f_a / (f_b - f_a)
        f_c = f(C, delta, rd, v, a, tau)
        if f_c * f_b < 0:
            A = B
            f_a = f_b
        else:
            f_a = f_a / 2
        B = C
        f_b = f_c
    return np.exp(A / 2.0)


def update_rates(home_team, away_team, xG_h, xG_a, table, h_adv=60.0 / 173.7178):
    rating_table = table.copy()

    h_att_rate = (rating_table.loc[home_team]['attacking rate'] - 1500.0) / 173.7178
    h_def_rate = (rating_table.loc[home_team]['defending rate'] - 1500.0) / 173.7178
    a_att_rate = (rating_table.loc[away_team]['attacking rate'] - 1500.0) / 173.7178
    a_def_rate = (rating_table.loc[away_team]['defending rate'] - 1500.0) / 173.7178
    h_rd_att = rating_table.loc[home_team]['RD_a'] / 173.7178
    h_rd_def = rating_table.loc[home_team]['RD_d'] / 173.7178
    a_rd_att = rating_table.loc[away_team]['RD_a'] / 173.7178
    a_rd_def = rating_table.loc[away_team]['RD_d'] / 173.7178
    h_vol_att = rating_table.loc[home_team]['vol_a']
    h_vol_def = rating_table.loc[home_team]['vol_d']
    a_vol_att = rating_table.loc[away_team]['vol_a']
    a_vol_def = rating_table.loc[away_team]['vol_d']

    v_h_att = 1.0 / (g(a_rd_def) * g(a_rd_def) * E(h_att_rate + h_adv, a_def_rate, a_rd_def) *
                     (1 - E(h_att_rate + h_adv, a_def_rate, a_rd_def)))
    v_h_def = 1.0 / (g(a_rd_att) * g(a_rd_att) * E(h_def_rate + h_adv, a_att_rate, a_rd_att) *
                     (1 - E(h_def_rate + h_adv, a_att_rate, a_rd_att)))
    v_a_att = 1.0 / (g(h_rd_def) * g(h_rd_def) * E(a_att_rate, h_def_rate + h_adv, h_rd_def) *
                     (1 - E(a_att_rate, h_def_rate + h_adv, h_rd_def)))
    v_a_def = 1.0 / (g(h_rd_att) * g(h_rd_att) * E(a_def_rate, h_att_rate + h_adv, h_rd_att) *
                     (1 - E(a_def_rate, h_att_rate + h_adv, h_rd_att)))

    exp_h_scored = E(h_att_rate + h_adv, a_def_rate, a_rd_def)
    exp_a_scored = E(a_att_rate, h_def_rate + h_adv, h_rd_def)
    real_h_scored = len(scale[scale < xG_h]) / 20000
    real_a_scored = len(scale[scale < xG_a]) / 20000

    delta_h_att = v_h_att * g(a_rd_def) * (real_h_scored - exp_h_scored)
    delta_h_def = v_h_def * g(a_rd_att) * (real_a_scored - exp_a_scored) * (-1)
    delta_a_att = v_a_att * g(h_rd_def) * (real_a_scored - exp_a_scored)
    delta_a_def = v_a_def * g(h_rd_att) * (real_h_scored - exp_h_scored) * (-1)

    h_vol_att = update_volatility(delta_h_att, h_rd_att, h_vol_att, v_h_att)
    h_vol_def = update_volatility(delta_h_def, h_rd_def, h_vol_def, v_h_def)
    a_vol_att = update_volatility(delta_a_att, a_rd_att, a_vol_att, v_a_att)
    a_vol_def = update_volatility(delta_a_def, a_rd_def, a_vol_def, v_a_def)

    h_rd_att = 1.0 / np.sqrt(1.0 / (np.power(h_rd_att, 2) + np.power(h_vol_att, 2)) + 1.0 / v_h_att)
    h_rd_def = 1.0 / np.sqrt(1.0 / (np.power(h_rd_def, 2) + np.power(h_vol_def, 2)) + 1.0 / v_h_def)
    a_rd_att = 1.0 / np.sqrt(1.0 / (np.power(a_rd_att, 2) + np.power(a_vol_att, 2)) + 1.0 / v_a_att)
    a_rd_def = 1.0 / np.sqrt(1.0 / (np.power(a_rd_def, 2) + np.power(a_vol_def, 2)) + 1.0 / v_a_def)

    h_att_rate = h_att_rate + (np.power(h_rd_att, 2) * delta_h_att) / v_h_att
    h_def_rate = h_def_rate + (np.power(h_rd_def, 2) * delta_h_def) / v_h_def
    a_att_rate = a_att_rate + (np.power(a_rd_att, 2) * delta_a_att) / v_a_att
    a_def_rate = a_def_rate + (np.power(a_rd_def, 2) * delta_a_def) / v_a_def

    rating_table['attacking rate'][home_team] = h_att_rate * 173.7178 + 1500
    rating_table['attacking rate'][away_team] = a_att_rate * 173.7178 + 1500
    rating_table['defending rate'][home_team] = h_def_rate * 173.7178 + 1500
    rating_table['defending rate'][away_team] = a_def_rate * 173.7178 + 1500

    rating_table['RD_a'][home_team] = h_rd_att * 173.7178
    rating_table['RD_a'][away_team] = a_rd_att * 173.7178
    rating_table['RD_d'][home_team] = h_rd_def * 173.7178
    rating_table['RD_d'][away_team] = a_rd_def * 173.7178

    rating_table['vol_a'][home_team] = h_vol_att
    rating_table['vol_a'][away_team] = a_vol_att
    rating_table['vol_d'][home_team] = h_vol_def
    rating_table['vol_d'][away_team] = a_vol_def

    table = rating_table

    return table, [exp_h_scored, exp_a_scored], [real_h_scored, real_a_scored]


def replace_teams(table, new_year):
    new_teams = teams[new_year - 2014]
    old_teams = np.array(table.index)
    up = []
    down = []
    avg_a = 0.0
    avg_d = 0.0
    for i in range(len(new_teams)):
        if new_teams[i] in old_teams:
            pass
        else:
            up.append(new_teams[i])
        if old_teams[i] in new_teams:
            pass
        else:
            down.append([old_teams[i], i])
            avg_a = avg_a + table['attacking rate'][old_teams[i]]
            avg_d = avg_d + table['defending rate'][old_teams[i]]
    avg_a = avg_a / len(down)
    avg_d = avg_d / len(down)
    table['RD_a'] = 150
    table['RD_d'] = 150
    table['vol_a'] = 0.06
    table['vol_d'] = 0.06
    for i in range(len(down)):
        table['attacking rate'][down[i][0]] = avg_a
        table['defending rate'][down[i][0]] = avg_d
        table['RD_a'][down[i][0]] = 350
        table['RD_d'][down[i][0]] = 350
        table['vol_a'][down[i][0]] = 0.06
        table['vol_d'][down[i][0]] = 0.06
        old_teams[down[i][1]] = up[i]
    table.set_index(old_teams, inplace=True)
    table.set_index(old_teams, inplace=True)
    return table


def run_train():
    time_series = {}
    for team in big_data_set['team_h'].unique():
        time_series[team] = []
    table = init_rates()
    for i in seasons[:-1]:
        res = big_data_set[big_data_set['season'] == int(i)]
        for j in res.index:
            table = update_rates(home_team=res['team_h'][j], away_team=res['team_a'][j],
                                 xG_h=res['xG_h'][j], xG_a=res['xG_a'][j], table=table)[0]
            if res['result'][j] == 'w':
                home_result = 'victory ('+ str(int(res['goals_h'][j]))+':'+ str(int(res['goals_a'][j])) +') against ' + res['team_a'][j]
                away_result = 'defeat ('+ str(int(res['goals_h'][j]))+':'+ str(int(res['goals_a'][j])) +') against ' + res['team_h'][j]
            elif res['result'][j] == 'd':
                home_result = 'draw (' + str(int(res['goals_h'][j])) + ':' + str(int(res['goals_a'][j])) + ') against ' + res['team_a'][j]
                away_result = 'draw (' + str(int(res['goals_h'][j])) + ':' + str(int(res['goals_a'][j])) + ') against ' + res['team_h'][j]
            else:
                home_result = 'defeat (' + str(int(res['goals_h'][j])) + ':' + str(int(res['goals_a'][j])) + ') against ' + res['team_a'][j]
                away_result = 'victory (' + str(int(res['goals_h'][j])) + ':' + str(int(res['goals_a'][j])) + ') against ' + res['team_h'][j]
            time_series[res['team_h'][j]].append([table['attacking rate'][res['team_h'][j]],
                                                  table['defending rate'][res['team_h'][j]],
                                                  res['index'][j] + 1, res['season'][j], res['datetime'][j],
                                                  res['id'][j], home_result])
            time_series[res['team_a'][j]].append([table['attacking rate'][res['team_a'][j]],
                                                  table['defending rate'][res['team_a'][j]],
                                                  res['index'][j] + 1, res['season'][j], res['datetime'][j],
                                                  res['id'][j], away_result])
        table = replace_teams(table, int(i) + 1)

    res = big_data_set[big_data_set['season'] == 2020]
    for j in res.index:
        table = update_rates(home_team=res['team_h'][j], away_team=res['team_a'][j],
                             xG_h=res['xG_h'][j], xG_a=res['xG_a'][j], table=table, h_adv=0)[0]
        if res['result'][j] == 'w':
            home_result = 'victory (' + str(int(res['goals_h'][j])) + ':' + str(int(res['goals_a'][j])) + ') against ' + res['team_a'][j]
            away_result = 'defeat (' + str(int(res['goals_h'][j])) + ':' + str(int(res['goals_a'][j])) + ') against ' + res['team_h'][j]
        elif res['result'][j] == 'd':
            home_result = 'draw (' + str(int(res['goals_h'][j])) + ':' + str(int(res['goals_a'][j])) + ') against ' + res['team_a'][j]
            away_result = 'draw (' + str(int(res['goals_h'][j])) + ':' + str(int(res['goals_a'][j])) + ') against ' + res['team_h'][j]
        else:
            home_result = 'defeat (' + str(int(res['goals_h'][j])) + ':' + str(int(res['goals_a'][j])) + ') against ' + res['team_a'][j]
            away_result = 'victory (' + str(int(res['goals_h'][j])) + ':' + str(int(res['goals_a'][j])) + ') against ' + res['team_h'][j]
        time_series[res['team_h'][j]].append([table['attacking rate'][res['team_h'][j]],
                                              table['defending rate'][res['team_h'][j]],
                                              res['index'][j] + 1, res['season'][j], res['datetime'][j],
                                              res['id'][j], home_result])
        time_series[res['team_a'][j]].append([table['attacking rate'][res['team_a'][j]],
                                              table['defending rate'][res['team_a'][j]],
                                              res['index'][j] + 1, res['season'][j], res['datetime'][j],
                                              res['id'][j], away_result])
    return table, time_series

curr_table, time_series = run_train()

def create_team_table(name):
    return pd.DataFrame(data = time_series[name], columns = ['attacking rate', 'defending rate', 'tour',
                                                             'season', 'date', 'match_id', 'result'])

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

for name in list(color_dict.keys()):
    create_team_table(name = name).to_csv('teams time series/'+name+' stats.csv')

curr_table.to_csv('current ratings.csv')