# This imports the client to scrape basketball-reference for NBA data
from basketball_reference_web_scraper import client
import pandas as pd

season_min = 2005
season_max = 2020


#load nba advanced stastics from basketball reference
nba_df = pd.DataFrame()
for year in range(season_min-1,season_max):
    temp = pd.DataFrame(i for i in client.players_advanced_season_totals(season_end_year=year))
    temp['year'] = year
    nba_df = nba_df.append(temp)
    
#define columns to be averaged and those to be summed
avg_columns = ['assist_percentage','block_percentage','defensive_rebound_percentage','free_throw_attempt_rate',
               'offensive_rebound_percentage','steal_percentage','three_point_attempt_rate','total_rebound_percentage',
               'true_shooting_percentage','turnover_percentage','usage_percentage']

sum_columns = ['minutes_played']

group_by_columns = ['name','year']

#merge partial season statistics together
#get minutes proportions
sum_df = nba_df.groupby(['name','year']).sum()[sum_columns].reset_index()
sum_df = sum_df.rename(columns = {'minutes_played':'total_minutes_played'})
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
exclude_list = [i for i in nba_df_final['name'][nba_df_final['year']==season_min-1]]
nba_df_final[~nba_df_final['name'].isin(exclude_list)]

#write to csv
nba_df_final.to_csv('data/nba_df.csv')



from sportsreference.ncaab.conferences import Conferences

#isolate only top conferences
conf_id = ['big-ten','big-12','big-east','pac-12','acc','sec','wcc','mwc','atlantic-10','wac']

#create list of team names
team_list = []
for i in conf_id:
    for i in Conferences.conferences[i]['teams'].values():
        team_list.append(i)

#this imports college bb stats from sports reference
from sportsreference.ncaab import teams

cc_df = pd.DataFrame()
years = range(season_min+3,season_max+1)
for year in years:
    for i in teams.Teams(year = year):
        if i.name in team_list: 
            roster = i.roster
            for player in roster.players:
                temp = player.dataframe
                temp['name'] = player.name
                cc_df = pd.concat([cc_df,temp])
            print('Added:',i.name)
    
#convert season into column/replace index
cc_df = cc_df.reset_index().rename(columns = {'level_0':'season'})
#remove any duplicates
cc_df = cc_df.drop_duplicates()

#add years played
cc_df = cc_df.sort_values(by=['name','season'],ascending = True)
cc_df['yrs_played'] = cc_df.groupby('name').cumcount()+1
exclude_list = [i for i in cc_df['name'][cc_df['season']=='2008-09']]
cc_df[~cc_df['name'].isin(exclude_list)]

#write to csv
cc_df.to_csv('data/cc_df.csv')