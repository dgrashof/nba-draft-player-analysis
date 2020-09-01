#create model data

import pandas as pd
import numpy as np
import os

os.chdir('C:\\Users\\David\\OneDrive')

#read college data
cc_df = pd.read_csv('Projects/nba-draft-player-analysis/data/cc_df.csv')
cc_df = cc_df.drop(columns = 'Unnamed: 0')
#max year in college per player excluding career
cc_df_max = pd.merge(cc_df,cc_df.groupby('player_id',as_index = False).agg({'yrs_played':np.max}),how = 'inner',on = ['player_id','yrs_played'])

#create per 40 min stats
cols = ['assists','blocks','defensive_rebounds','free_throw_attempts','free_throws','offensive_rebounds','personal_fouls',
       'points','steals','three_point_attempts','three_pointers','total_rebounds','turnovers','two_point_attempts',
       'two_pointers']

for i in cols:
    cc_df_max[i+'_per_40'] = round((cc_df_max[i]/cc_df_max['minutes_played'])*40,2)

cc_df_max.drop(cols,axis = 1,inplace = True)

#read nba data
nba_df = pd.read_csv('Projects/nba-draft-player-analysis/data/nba_df.csv')
nba_df = nba_df.drop(columns = 'Unnamed: 0')

#extract first n years of data in NBA
players_4_yrs = [i for i in nba_df['name'][nba_df['yrs_played']==4]]
three_year_df = nba_df[(nba_df['yrs_played']<=4)&(nba_df['name'].isin(players_4_yrs))]
three_year_df_final = three_year_df.groupby('col_lkup').agg(np.max)['value_over_replacement_player']
three_year_df_final = three_year_df_final.reset_index()
three_year_df_final.rename(columns = {'value_over_replacement_player':'nba_vorp'},inplace = True)

#merge 3-year VORP with max year college data
combo_df = pd.merge(left = cc_df_max,right=three_year_df_final,how = 'inner',on = 'col_lkup')

#read hs rankings
#hs_df = pd.read_csv('Projects/nba-draft-player-analysis/data/hs_rankings_df.csv')
#hs_df.hs_recruit_ranking = hs_df.hs_recruit_ranking.astype('int64').astype(str)
#merge hs recruiting rankings with career data
#combo_df = pd.merge(left = combo_df,right=hs_df,how = 'left',on = 'name')
#combo_df.hs_recruit_ranking = combo_df.hs_recruit_ranking.fillna('unranked')

#load combine data
#combine_df = pd.read_csv('Projects/nba-draft-player-analysis/data/combine_df.csv')
#combine_df = combine_df.add_prefix('combine_')
#combo_df = pd.merge(left = combo_df,right = combine_df,how = 'left',left_on = 'name',right_on = 'combine_name')

#cleanup data
#combo_df = combo_df.drop(columns = 'combine_name')
combo_df.columns = map(str.lower, combo_df.columns)

#write model_data
combo_df.to_csv('C:/Users/David/OneDrive/Projects/nba-draft-player-analysis/data/modeling_data.csv',index=False)


####################
#classification data
####################


os.chdir('C:\\Users\\David\\OneDrive')

#read college data
cc_df = pd.read_csv('Projects/nba-draft-player-analysis/data/cc_df.csv')
cc_df = cc_df.drop(columns = 'Unnamed: 0')
#max year in college per player excluding career
cc_df_max = pd.merge(cc_df,cc_df.groupby('player_id',as_index = False).agg({'yrs_played':np.max}),how = 'inner',on = ['player_id','yrs_played'])

#create per 40 min stats
cols = ['assists','blocks','defensive_rebounds','free_throw_attempts','free_throws','offensive_rebounds','personal_fouls',
       'points','steals','three_point_attempts','three_pointers','total_rebounds','turnovers','two_point_attempts',
       'two_pointers']

for i in cols:
    cc_df_max[i+'_per_40'] = round((cc_df_max[i]/cc_df_max['minutes_played'])*40,2)

cc_df_max.drop(cols,axis = 1,inplace = True)

#read nba data
nba_df = pd.read_csv('Projects/nba-draft-player-analysis/data/nba_df.csv')
nba_df = nba_df.drop(columns = 'Unnamed: 0')

#extract max mins in four yrs of NBA
players_4_yrs = [i for i in nba_df['name'][nba_df['yrs_played']==4]]
nba_df = nba_df[(nba_df['yrs_played']<=4)&(nba_df['name'].isin(players_4_yrs))]
mp_df_final = nba_df.groupby('col_lkup').agg({'minutes_played':np.max})
mp_df_final = mp_df_final.reset_index()
mp_df_final.rename(columns = {'minutes_played':'max_mp'},inplace = True)
mp_df_final.to_csv('test.csv')


#merge 4-year max MP with max year college data
combo_df = pd.merge(left = cc_df_max,right=mp_df_final,how = 'inner',on = 'col_lkup')

#extract all-stars
as_df = pd.read_excel('Projects/nba-draft-player-analysis/data/awards.xlsx',sheet_name = "all_star",usecols=[0,1])
as_df = as_df.groupby('name').agg({'year':np.min})
as_df = as_df.reset_index()
as_df['all_star_flag'] = '1'
#merge all star year
as_df_final = pd.merge(left=nba_df,right=as_df,how = 'inner',on=['name','year'])
as_df_final
as_df_final = as_df_final[['name','year','all_star_flag']]
#merge all-star flag to combo_df
combo_df = pd.merge(left=combo_df,right=as_df_final,how = 'left',on = 'name')
combo_df.all_star_flag = combo_df.all_star_flag.fillna('0')

combo_df.columns = map(str.lower, combo_df.columns)


combo_df.loc[combo_df['all_star_flag'] == '1','class'] = 'all_star'
combo_df.loc[(combo_df['max_mp'] > 2200) & (combo_df['all_star_flag'] != '1'),'class'] = 'starter'
combo_df.loc[(combo_df['max_mp'] > 1300) & (combo_df['max_mp']<=2200) & (combo_df['all_star_flag'] != '1'),'class'] = 'role_player'
combo_df.loc[(combo_df['max_mp'] <= 1300) & (combo_df['all_star_flag'] != '1'),'class'] = 'scrub'


#write model_data
combo_df.to_csv('C:/Users/David/OneDrive/Projects/nba-draft-player-analysis/data/modeling_data_classification.csv',index=False)