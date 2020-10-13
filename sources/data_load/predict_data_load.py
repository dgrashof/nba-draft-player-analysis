from sportsreference.ncaab.roster import Player
from datetime import datetime
import pandas as pd 
import os
import numpy as np
os.chdir('C:\\Users\\David\\OneDrive\Projects\\nba-draft-player-analysis')

player_lkup = pd.read_csv('data\player_lkup.csv')

player = [i for i in player_lkup['Player']]

cc_df = pd.DataFrame()

for j in [1,2,3,4,5]:
    for i in player:
        try:
            temp = Player(i.split()[0].lower()+'-'+i.split()[1].lower()+'-'+str(j)).dataframe
            temp['name'] = [i]*len(temp)
            cc_df = cc_df.append(temp)
        except:
            pass

def season_calc(x):
    if x == 'Career':
        return('Career')
    else:
        return(x[:4])


cc_df.reset_index(inplace=True)
cc_df['season'] = cc_df.level_0.map(season_calc)

cc_df_max = cc_df[cc_df['season']!='Career']
cc_df = cc_df[cc_df['season']=='Career']


#calculate years played
cc_df_max = cc_df_max.sort_values(by=['player_id','season'],ascending = True)
cc_df_max['yrs_played'] = cc_df_max.groupby('player_id').cumcount()+1
cc_df = cc_df.drop(columns = 'level_0')

cc_df = pd.merge(cc_df,cc_df_max.groupby('player_id',as_index = False).agg({'yrs_played':np.max}),how = 'inner',on = ['player_id'])
cc_df = cc_df.dropna(how='any')

#create per 40 min stats
cols = ['assists','blocks','defensive_rebounds','free_throw_attempts','free_throws','offensive_rebounds','personal_fouls',
       'points','steals','three_point_attempts','three_pointers','total_rebounds','turnovers','two_point_attempts',
       'two_pointers']

for i in cols:
    cc_df[i+'_per_40'] = round((cc_df[i].astype('int64')/cc_df['minutes_played'].astype('int64'))*40,2)

#cc_df.drop(cols,axis = 1,inplace = True)


cc_df.to_csv('data\predict_2018.csv',index=False)