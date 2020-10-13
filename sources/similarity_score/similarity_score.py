#load data
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

#training data
train_prelim = pd.read_csv('C:/Users/David/OneDrive/Projects/nba-draft-player-analysis/data/modeling_data.csv').dropna(how='any').reset_index(drop=True)

#predict data
predict_prelim = pd.read_csv('C:/Users/David/OneDrive/Projects/nba-draft-player-analysis/data/predict_2018.csv')

cols = ['assists_per_40','blocks_per_40','free_throw_attempts_per_40','free_throw_percentage','personal_fouls_per_40',
        'points_per_40','steals_per_40','three_point_attempts_per_40','three_point_percentage','two_pointers_per_40',
        'two_point_percentage','assists_per_40','total_rebounds_per_40']

groupby = 'yrs_played'

scaler = StandardScaler()

train = pd.DataFrame(scaler.fit_transform(train_prelim[cols]),columns = cols)
train = pd.concat([train,train_prelim[['col_lkup','yrs_played']]],axis=1)

predict = pd.DataFrame(scaler.transform(predict_prelim[cols]),columns = cols)
predict = pd.concat([predict,predict_prelim[['name','yrs_played']]],axis=1)

name = []
comp_name = []
score = []

for i in range(0,len(predict)):
    comp = predict.iloc[i]
    comp_2 = train[train['yrs_played']==int(comp[groupby])]
    for j in range(0,len(comp_2)):
        name.append(comp['name'])
        comp_name.append(comp_2['col_lkup'].iloc[j])
        score.append(round(float(cosine_similarity(np.array(comp_2.reset_index().iloc[j][cols]).reshape(1,-1),np.array(comp[cols]).reshape(1,-1))),2))

results = pd.DataFrame({'name':name,'comp':comp_name,'score':score}).sort_values(by=['name','score'],ascending=[True,False])
results['rank'] = results.groupby('name').cumcount()+1
results = results[results['rank']<=5]
#results = pd.merge(results,train_prelim[['col_lkup','nba_vorp']],
#                  how='left',left_on='comp',right_on='col_lkup').drop(columns='col_lkup')
results.to_csv('similarity_score_2018.csv',index = False)