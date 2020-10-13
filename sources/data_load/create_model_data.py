#create individual stat model data

import pandas as pd
import numpy as np
import os

os.chdir('C:\\Users\\David\\OneDrive\\Projects\\nba-draft-player-analysis')

#####################
####read college data
#####################

cc_df = pd.read_csv('data/cc_df.csv')
cc_df['season'] = cc_df['level_0'].str[:4]
cc_df = cc_df.drop(columns = ['Unnamed: 0','level_0'])

def weighted_avg(df,index,value,weight):
   temp_df = df[df[weight]>0]
   weighted_df = temp_df.groupby(index).apply(lambda x: np.average(x[value], weights=x[weight])).reset_index()
   weighted_df = pd.merge(left=weighted_df,right=temp_df.groupby(index).agg({weight:np.sum}).reset_index(),how='left',on='col_lkup')
   weighted_df = weighted_df.rename(columns={0:str(value)})
   weighted_df.drop(columns=[weight],inplace =True)
   return(weighted_df)

#get career averages for three point % and free throw %
weighted_df = weighted_avg(cc_df,'col_lkup','free_throw_percentage','free_throw_attempts')
weighted_df = pd.merge(weighted_df,weighted_avg(cc_df,'col_lkup','three_point_percentage','three_point_attempts'),how='left',on='col_lkup')
weighted_df = pd.merge(weighted_df,weighted_avg(cc_df,'col_lkup','two_point_percentage','two_point_attempts'),how='left',on='col_lkup')
weighted_df = pd.merge(weighted_df,cc_df.groupby('col_lkup').yrs_played.max().reset_index(),how='left',on='col_lkup')
weighted_df = pd.merge(weighted_df,cc_df.groupby('col_lkup').season.max().reset_index(),how='left',on='col_lkup').rename(columns={'season':'max_season'})

cc_df_sum = cc_df.groupby(['col_lkup','height','position']).agg({'steals':'sum','blocks':'sum','minutes_played':'sum','personal_fouls':'sum','assists':'sum','free_throw_attempts':'sum','free_throws':'sum','offensive_rebounds':'sum','defensive_rebounds':'sum','points':'sum','total_rebounds':'sum','turnovers':'sum','two_point_attempts':'sum','two_pointers':'sum','three_point_attempts':'sum','three_pointers':'sum'}).reset_index()
cc_df_sum = pd.merge(cc_df_sum,cc_df.groupby('col_lkup').yrs_played.max().reset_index(),how='left',on='col_lkup')


#create per 40 min stats
cols = ['steals','blocks','personal_fouls','assists','free_throw_attempts','free_throws','offensive_rebounds','defensive_rebounds','points','total_rebounds','turnovers','two_point_attempts','two_pointers','three_point_attempts','three_pointers']

for i in cols:
    cc_df_sum[i+'_per_40'] = round((cc_df_sum[i]/cc_df_sum['minutes_played'])*40,2)

#cc_df_sum.drop(cols,axis = 1,inplace = True)
cc_df_final = pd.merge(cc_df_sum,weighted_df,how='inner',on=['col_lkup','yrs_played'])

################
##read nba data
################

nba_df = pd.read_csv('data/nba_df_old.csv')
nba_df = nba_df.drop(columns = 'Unnamed: 0')

#extract first n years of data in NBA
players_4_yrs = [i for i in nba_df['name'][nba_df['yrs_played']==4]]
four_year_df = nba_df[(nba_df['yrs_played']<=4)&(nba_df['name'].isin(players_4_yrs))]


weighted_df = weighted_avg(df=four_year_df,index='col_lkup',value='three_point_percentage',weight='attempted_three_point_field_goals')
weighted_df = pd.merge(weighted_df,weighted_avg(four_year_df,'col_lkup',value='free_throw_percentage',weight='attempted_free_throws'),how='left',on='col_lkup')
weighted_df = pd.merge(weighted_df,weighted_avg(four_year_df,'col_lkup',value='assist_percentage',weight='minutes_played'),how='left',on='col_lkup')
weighted_df = pd.merge(weighted_df,weighted_avg(four_year_df,'col_lkup',value='block_percentage',weight='minutes_played'),how='left',on='col_lkup')
weighted_df = pd.merge(weighted_df,weighted_avg(four_year_df,'col_lkup',value='steal_percentage',weight='minutes_played'),how='left',on='col_lkup')
weighted_df = pd.merge(weighted_df,weighted_avg(four_year_df,'col_lkup',value='usage_percentage',weight='minutes_played'),how='left',on='col_lkup')
weighted_df = pd.merge(weighted_df,weighted_avg(four_year_df,'col_lkup',value='total_rebound_percentage',weight='minutes_played'),how='left',on='col_lkup')
weighted_df = pd.merge(weighted_df,weighted_avg(four_year_df,'col_lkup',value='true_shooting_percentage',weight='minutes_played'),how='left',on='col_lkup')
weighted_df = pd.merge(weighted_df,weighted_avg(four_year_df,'col_lkup',value='turnover_percentage',weight='minutes_played'),how='left',on='col_lkup')

four_year_df_sum = four_year_df.groupby('col_lkup').agg({'value_over_replacement_player':'sum','steals':'sum','blocks':'sum','minutes_played':'sum','assists':'sum','turnovers':'sum','attempted_three_point_field_goals':'sum','attempted_free_throws':'sum'}).reset_index()

#create per 40 min stats
cols = ['steals','blocks','assists','turnovers','attempted_three_point_field_goals','attempted_free_throws']

for i in cols:
    four_year_df_sum[i+'_per_40'] = round((four_year_df_sum[i]/four_year_df_sum['minutes_played'])*40,2)


four_year_df_final = pd.merge(four_year_df_sum,weighted_df,how='inner',on='col_lkup')
four_year_df_final = four_year_df_final.add_prefix('nba_')
four_year_df_final.rename(columns={'nba_col_lkup':'col_lkup'},inplace=True)

#merge 3-year VORP with max year college data
combo_df = pd.merge(left = cc_df_final,right=four_year_df_final,how = 'inner',on = 'col_lkup')


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
combo_df.columns = map(str.lower, combo_df.columns)

#write model_data
combo_df.to_csv('data/modeling_data.csv',index=False)