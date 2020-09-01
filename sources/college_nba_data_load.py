# This imports the client to scrape basketball-reference for NBA data
from basketball_reference_web_scraper import client
from datetime import datetime
import pandas as pd 
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process
import os
os.chdir('C:\\Users\\David\\OneDrive')

season_min = 1989
season_max = 2020

print('Downloading NBA DF',datetime.now().strftime("%d-%b-%Y [%H:%M:%S.%f]"))

#load nba advanced stastics from basketball reference
nba_df = pd.DataFrame()
for year in range(season_min,season_max):
    temp = pd.DataFrame(i for i in client.players_advanced_season_totals(season_end_year=year))
    temp['year'] = year
    nba_df = nba_df.append(temp)

#define columns to be averaged and those to be summed
avg_columns = ['assist_percentage','block_percentage','defensive_rebound_percentage','free_throw_attempt_rate',
               'offensive_rebound_percentage','steal_percentage','three_point_attempt_rate','total_rebound_percentage',
               'true_shooting_percentage','turnover_percentage','usage_percentage','box_plus_minus']

sum_columns = ['minutes_played','value_over_replacement_player']

group_by_columns = ['name','year']

#merge partial season statistics together
#get minutes proportions
sum_df = nba_df.groupby(['name','year']).sum()[sum_columns].reset_index()
sum_df = sum_df.rename(columns = {'minutes_played':'total_minutes_played'})
nba_df = nba_df.drop(columns = 'value_over_replacement_player')
nba_df = pd.merge(left = nba_df,right = sum_df,how = 'left',on = ['name','year'])
nba_df['prop_mins'] = round(nba_df['minutes_played']/nba_df['total_minutes_played'],2)

#recalculate weighted minutes for rows with partial season data
def prop_calc(col_1,col_2,df):
    df['rev_'+col_1] = df[col_1] * df[col_2]
    df.drop(columns = col_1,inplace = True)
    df[col_1] = df['rev_'+col_1]
    df.drop(columns = 'rev_'+col_1,inplace = True)

#apply calcuation to df and drop old column values
for i in avg_columns:
    prop_calc(i,'prop_mins',nba_df)

#merge partial rows together
nba_df_final = nba_df.groupby(['name','year']).agg(sum).reset_index()[avg_columns+sum_columns+group_by_columns]
#tack on age
age_lkup = nba_df[['name','year','age']].drop_duplicates()
nba_df_final = pd.merge(left=nba_df_final,right=age_lkup,on = ['name','year'],how = 'left')
#calculate years plated
nba_df_final = nba_df_final.sort_values(by=['name','year'],ascending = True)
nba_df_final['yrs_played'] = nba_df_final.groupby('name').cumcount()+1
"""
Since I set an arbitrary starting season for for my dataset I run into the issue of having players who
are wrongly labeled as being in their first season. Therefore, I want to remove all players who played in the first 
season from all the seasons
"""
exclude_list = [i for i in nba_df_final['name'][nba_df_final['year']==season_min]]
nba_df_final = nba_df_final[~nba_df_final['name'].isin(exclude_list)]
nba_df_final.to_csv('Projects/nba-draft-player-analysis/data/prelim_nba_df.csv')

#load college data
college_lkup = pd.read_csv('Projects/nba-draft-player-analysis/data/draft_hist_df.csv',encoding='latin1')

y = list(college_lkup['name'])
x = list(nba_df['name'].drop_duplicates())

def checker(wrong_options,correct_options):
    names_array=[]
    ratio_array=[]    
    for wrong_option in wrong_options:
        if wrong_option in correct_options:
            names_array.append(wrong_option)
            ratio_array.append('100')
        else:   
            x=process.extractOne(wrong_option,correct_options,scorer=fuzz.partial_ratio)
            names_array.append(x[0])
            ratio_array.append(x[1])
    return(names_array,ratio_array)

name_match,ratio_match=checker(x,y)
df1 = pd.DataFrame()
df1['name']=pd.Series(x)
df1['correct_names']=pd.Series(name_match)
df1['correct_ratio']=pd.Series(ratio_match).astype('int64')
df1 = pd.merge(df1,college_lkup[['name','college']].rename(columns={'name':'correct_names'}),how = 'left',on = 'correct_names')
print(nba_df_final.shape)
nba_df_final = pd.merge(nba_df_final,df1[df1['correct_ratio']>80][['name','college']],how = 'inner',on = 'name')
print(nba_df_final.shape)
nba_df_final['lkup'] = nba_df_final['name']+" "+nba_df_final['college']
print('Finished Downloading NBA DF/Starting CC DF',datetime.now().strftime("%d-%b-%Y [%H:%M:%S.%f]"))

from sportsreference.ncaab.roster import Player

player = [i for i in nba_df_final['name'].drop_duplicates()]

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

#exception
temp = Player('dennis-smithjr-1').dataframe
temp['name'] = ['Dennis Smith Jr.']*len(temp)
cc_df = cc_df.append(temp)

cc_df.reset_index(inplace=True)
cc_df['season'] = cc_df.level_0.map(season_calc)

#calculate years plated
cc_df = cc_df.sort_values(by=['player_id','season'],ascending = True)
cc_df['yrs_played'] = cc_df.groupby('player_id').cumcount()+1
cc_df = cc_df[cc_df['season']!='Career']
cc_df = cc_df[cc_df['season'].astype('int64')>min(nba_df.year-5)]
#lookup
cc_df['col_lkup'] = cc_df['name']+" "+cc_df['team_abbreviation']

print('Finished CC DF/Starting Fuzzy Match',datetime.now().strftime("%d-%b-%Y [%H:%M:%S.%f]"))

x = list(set(nba_df_final[nba_df_final['college']!='None']['lkup'].to_list()))
y = list(set(cc_df['col_lkup'].to_list()))

def checker(wrong_options,correct_options):
    names_array=[]
    correct_names_array=[]
    ratio_array=[]    
    for wrong_option in wrong_options:
        if wrong_option in correct_options:
            names_array.append(wrong_option)
            correct_names_array.append(wrong_option)
            ratio_array.append('100')
        else:   
            orig_name = ' '.join(wrong_option.split(" ")[:2])
            x=process.extract(orig_name,correct_options,scorer=fuzz.partial_ratio)
            name,value = zip(*x)
            y=process.extractOne(wrong_option,name,scorer=fuzz.partial_ratio)
            names_array.append(wrong_option)
            correct_names_array.append(y[0])
            ratio_array.append(y[1])
    return(names_array,correct_names_array,ratio_array)

original_name,name_match,ratio_match=checker(x,y)
df1 = pd.DataFrame()
df1['lkup']=pd.Series(original_name)
df1['col_lkup']=pd.Series(name_match)
df1['correct_ratio']=pd.Series(ratio_match).astype('int64')
df1 = df1[df1['correct_ratio']>=75]

#write matching dfs
pd.merge(nba_df_final,df1[['lkup','col_lkup']],how='inner',on='lkup').to_csv('Projects/nba-draft-player-analysis/data/nba_df.csv')
pd.merge(cc_df,df1[['col_lkup']],how='inner',on='col_lkup').to_csv('Projects/nba-draft-player-analysis/data/cc_df.csv')

print('Finished Fuzzy Match',datetime.now().strftime("%d-%b-%Y [%H:%M:%S.%f]"))