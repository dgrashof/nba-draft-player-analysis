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

#read nba data
nba_df = pd.read_csv('Projects/nba-draft-player-analysis/data/nba_df.csv')
nba_df = nba_df.drop(columns = 'Unnamed: 0')

#extract first n years of data in NBA
players_3_yrs = [i for i in nba_df['name'][nba_df['yrs_played']==3]]
three_year_df = nba_df[(nba_df['yrs_played']<=3)&(nba_df['name'].isin(players_3_yrs))]
three_year_df_final = three_year_df.groupby('col_lkup').agg(np.mean)['value_over_replacement_player']
three_year_df_final = three_year_df_final.reset_index()
three_year_df_final.rename(columns = {'value_over_replacement_player':'nba_vorp'},inplace = True)

#merge 3-year VORP with max year college data
combo_df = pd.merge(left = cc_df_max,right=three_year_df_final,how = 'inner',on = 'col_lkup')

#read hs rankings
hs_df = pd.read_csv('Projects/nba-draft-player-analysis/data/hs_rankings_df.csv')
hs_df.hs_recruit_ranking = hs_df.hs_recruit_ranking.astype('int64').astype(str)
#merge hs recruiting rankings with career data
combo_df = pd.merge(left = combo_df,right=hs_df,how = 'left',on = 'name')
combo_df.hs_recruit_ranking = combo_df.hs_recruit_ranking.fillna('unranked')

#load combine data
combine_df = pd.read_csv('Projects/nba-draft-player-analysis/data/combine_df.csv')
combine_df = combine_df.add_prefix('combine_')
combo_df = pd.merge(left = combo_df,right = combine_df,how = 'left',left_on = 'name',right_on = 'combine_name')

#cleanup data
combo_df = combo_df.drop(columns = 'combine_name')
combo_df.columns = map(str.lower, combo_df.columns)

#write model_data
combo_df.to_csv('C:/Users/David/OneDrive/Projects/nba-draft-player-analysis/data/modeling_data.csv',index=False)