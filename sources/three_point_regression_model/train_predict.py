import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
import seaborn as sns
from scipy.stats import beta as bt
import matplotlib.pyplot as  plt
import os
import warnings
warnings.filterwarnings("ignore")

train = 'no'
predict = 'yes'

train_file = 'data\\three_point_modeling_data.csv'
predict_file = 'data\\predict_2018.csv'


os.chdir('C:\\Users\\David\\OneDrive')
from utils.ml_data_proc import tts,model_predict_reg
from utils.graphs import regression_results
os.chdir('C:\\Users\\David\\OneDrive\\Projects\\nba-draft-player-analysis')

cont_fields = ['rev_three_point_percentage','three_point_attempts_per_40','rev_free_throw_percentage']
cat_fields = ['yrs_played']

response = 'rev_nba_three_point_percentage'

if train == 'yes':
    #load training data
    df = pd.read_csv(train_file)
    df = df.set_index('col_lkup')
    
    df['nba_three_point_made'] = round(df['nba_three_point_attempts']*df['nba_three_point_percentage'],0)
    df['three_point_made'] = round(df['three_point_attempts']*df['weighted_three_point_percentage'],0)
    df['free_throws_made'] = round(df['free_throw_attempts']*df['weighted_free_throw_percentage'],0)
    df = df[df['three_point_attempts_per_40']!=np.inf]
    
    def apply_ab(column_num,column_den,ab):
        new_column = (column_num+ab[0])/(column_den+ab[0]+ab[1])
        return(new_column)

    df_rev = pd.DataFrame()

    for pos,ab,cols in zip(['Forward','Guard','Center'],[(6,20),(24,51),(4,31)],[('nba_three_point_made','nba_three_point_attempts')]*3):
        temp = df[df['position']==pos]
        temp['rev_nba_three_point_percentage']=apply_ab(np.array(temp[cols[0]]),np.array(temp[cols[1]]),ab)
        df_rev = df_rev.append(temp)

    df_rev_ii = pd.DataFrame()

    for pos,ab,cols in zip(['Forward','Guard','Center'],[(36,83),(21,39),(16,52)],[('three_point_made','three_point_attempts')]*3):
        temp = df[df['position']==pos]
        temp['rev_three_point_percentage']=apply_ab(np.array(temp[cols[0]]),np.array(temp[cols[1]]),ab)
        df_rev_ii = df_rev_ii.append(temp)

    df_rev['rev_three_point_percentage'] = df_rev_ii['rev_three_point_percentage']

    df_rev_iii = pd.DataFrame()

    for pos,ab,cols in zip(['Forward','Guard','Center'],[(203,86),(211,68),(214,110)],[('free_throws_made','free_throw_attempts')]*3):
        temp = df[df['position']==pos]
        temp['rev_free_throw_percentage']=apply_ab(np.array(temp[cols[0]]),np.array(temp[cols[1]]),ab)
        df_rev_iii = df_rev_iii.append(temp)

    df_rev['rev_free_throw_percentage'] = df_rev_iii['rev_free_throw_percentage']
    
    os.chdir('C:\\Users\\David\\OneDrive\\Projects\\nba-draft-player-analysis\\sources\\three_point_regression_model')
    
    Xtrain,Xtest,ytrain,ytest = tts(df_rev,cat_fields,cont_fields,response)
    
    os.chdir('C:\\Users\\David\\OneDrive\\Projects\\nba-draft-player-analysis')
    
    with open('sources\\three_point_regression_model\\three_point_cols.txt', 'w') as f:
        for item in Xtrain.columns:
            f.write("%s\n" % item)
    
    # Train Models
    # =============================================================================
    # 1.1 | Linear Regression
    # =============================================================================
    
    model1 = LinearRegression()
    model1.fit(Xtrain,ytrain)
    y_predict = model1.predict(Xtest)
    print("Linear Regression Results")
    regression_results(y_predict,ytest)
    
    # =============================================================================
    # 1.2 | Random Forest Regression
    # =============================================================================
    
    model2 = RandomForestRegressor()
    model2 = RandomForestRegressor(random_state=42)
    model2.fit(Xtrain,ytrain)
    y_predict = model2.predict(Xtest)
    print("Random Forest Regression Results")
    regression_results(y_predict,ytest)
    
    # =============================================================================
    # 1.3 | SVR
    # =============================================================================
    
    model3 = SVR()
    model3.fit(Xtrain,ytrain)
    y_predict = model3.predict(Xtest)
    regression_results(y_predict,ytest)
    
    # =============================================================================
    # 1.4 | Stacking Regressor
    # =============================================================================
    
    #embed models in Classifier and use Logistic Regression as Final Estimator
    stacking_model = StackingRegressor(
    estimators=[("lr",model1),("rf",model2),("svr",model3)],final_estimator = LinearRegression())

    stacking_model.fit(Xtrain,ytrain)
    stacking_model_predict = stacking_model.predict(Xtest)
    regression_results(stacking_model_predict,ytest)
    
    # Serialize model to load for future predictions/avoid re-training
    pickle.dump(stacking_model, open('sources\\three_point_regression_model\\model_3pt_reg.sav', 'wb'))

if predict == 'yes':
    #Load Data
    df = pd.read_csv(predict_file)
    df = df.set_index('name')
    
    df['three_point_made'] = round(df['three_point_attempts']*df['three_point_percentage'],0)
    df['free_throws_made'] = round(df['free_throw_attempts']*df['free_throw_percentage'],0)
    df = df[df['three_point_attempts_per_40']!=np.inf]
    
    def apply_ab(column_num,column_den,ab):
        new_column = (column_num+ab[0])/(column_den+ab[0]+ab[1])
        return(new_column)

    df_rev = pd.DataFrame()

    for pos,ab,cols in zip(['Forward','Guard','Center'],[(36,83),(21,39),(16,52)],[('three_point_made','three_point_attempts')]*3):
        temp = df[df['position']==pos]
        temp['rev_three_point_percentage']=apply_ab(np.array(temp[cols[0]]),np.array(temp[cols[1]]),ab)
        df_rev = df_rev.append(temp)

    df_rev_ii = pd.DataFrame()

    for pos,ab,cols in zip(['Forward','Guard','Center'],[(203,86),(211,68),(214,110)],[('free_throws_made','free_throw_attempts')]*3):
        temp = df[df['position']==pos]
        temp['rev_free_throw_percentage']=apply_ab(np.array(temp[cols[0]]),np.array(temp[cols[1]]),ab)
        df_rev_ii = df_rev_ii.append(temp)

    df_rev['rev_free_throw_percentage'] = df_rev_ii['rev_free_throw_percentage']
    
    
    
    probs = model_predict_reg(df=df_rev,cat_fields=cat_fields,cont_fields=cont_fields,scaler='sources\\three_point_regression_model\\scaler.sav',encoder='sources\\three_point_regression_model\\encoder.sav'
                              ,model='sources\\three_point_regression_model\\model_3pt_reg.sav')
    
    results = probs.set_index(df_rev.index)
    results.to_csv('predictions\\results_2018_three_point.csv')
    
