from py_ball import draft
import pandas as pd
import numpy as np
import time

HEADERS = {'Connection': 'keep-alive',
           'Host': 'stats.nba.com',
           'Origin': 'http://stats.nba.com',
           'Upgrade-Insecure-Requests': '1',
           'Referer': 'stats.nba.com',
           'x-nba-stats-origin': 'stats',
           'x-nba-stats-token': 'true',
           'Accept-Language': 'en-US,en;q=0.9',
           "X-NewRelic-ID": "VQECWF5UChAHUlNTBwgBVw==",
           'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6)' +\
                         ' AppleWebKit/537.36 (KHTML, like Gecko)' + \
                         ' Chrome/81.0.4044.129 Safari/537.36'}

draft_df = pd.DataFrame()
cols = ['PLAYER_ID','PLAYER_NAME','SEASON','WINGSPAN','HEIGHT_W_SHOES','STANDING_REACH','BODY_FAT_PCT','HAND_LENGTH','HAND_WIDTH']

league_id = '00' #NBA
for i in [str(i)+'-'+str(i+1)[2:] for i in range(2000,2020)]:
    time.sleep(5)
    draft_data = draft.Draft(headers=HEADERS,
                         endpoint='draftcombinestats',
                         league_id=league_id,
                         season_year=i)
    draft_df = draft_df.append(pd.DataFrame(draft_data.data['DraftCombineStats'])[cols])
    print('Finished Combine Data: ',i)

draft_cols = ['PERSON_ID','PLAYER_NAME','SEASON','OVERALL_PICK','ORGANIZATION_TYPE','ORGANIZATION']
draft_history = draft.Draft(headers=HEADERS,
                            endpoint='drafthistory',
                            league_id=league_id,
                            season_year=i)
draft_hist_df = pd.DataFrame(draft_history.data['DraftHistory'])
draft_hist_df = draft_hist_df[draft_hist_df['SEASON'].astype('int64')>=2000]
draft_hist_df = draft_hist_df[draft_cols].rename(columns = {'PERSON_ID':'PLAYER_ID'})
draft_hist_df.OVERALL_PICK = draft_hist_df.OVERALL_PICK.astype('int64')
draft_df = pd.merge(left=draft_hist_df,right=draft_df,how = 'outer',on = ['PLAYER_ID','SEASON'])
draft_df['PLAYER_NAME'] = [x if len(x) > 3 else y for x,y in zip(draft_df['PLAYER_NAME_x'].astype(str),draft_df['PLAYER_NAME_y'].astype(str))]
draft_df = draft_df.drop(columns = ['PLAYER_NAME_x','PLAYER_NAME_y'])
draft_df = draft_df.drop(columns = ['PLAYER_ID'])
draft_df.columns = map(str.lower, draft_df.columns)
draft_df.to_csv('C:/Users/David/OneDrive/Projects/nba-draft-player-analysis/data/combine_df_revised.csv',index=False)