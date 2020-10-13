import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
import warnings
warnings.filterwarnings("ignore")

train_steals = 'yes'
predict_steals = 'yes'

train_blocks = 'yes'
predict_blocks = 'yes'

train_file = 'data\\modeling_data.csv'
predict_file = 'data\\predict_2020.csv'


os.chdir('C:\\Users\\David\\OneDrive')
from utils.ml_data_proc import tts,model_predict_reg
from utils.graphs import regression_results
os.chdir('C:\\Users\\David\\OneDrive\\Projects\\nba-draft-player-analysis')

#Steal Fields
stl_cont_fields = ['steals_per_40','usg_per_40','personal_fouls_per_40']
stl_cat_fields = ['yrs_played']
stl_response = 'nba_steals_per_40'

#Block Fields
blk_cont_fields = ['blocks_per_40','usg_per_40','personal_fouls_per_40','height']
blk_cat_fields = ['yrs_played']
blk_response = 'nba_blocks_per_40'


if train_steals == 'yes' or train_blocks =='yes':
    #load training data
    df = pd.read_csv(train_file)
    df = df.set_index('col_lkup')
    df = df[df['nba_minutes_played']>1000]
    df['usg_per_40'] = df['two_pointers_per_40']+df['three_point_attempts_per_40']+df['free_throw_attempts_per_40']

    
    os.chdir('C:\\Users\\David\\OneDrive\\Projects\\nba-draft-player-analysis\\sources\\three_point_regression_model')
    
    def train_models(Xtrain,Xtest,ytrain,ytest,model_name):
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
        pickle.dump(stacking_model, open(model_name+'.sav', 'wb'))
    
    if train_steals == 'yes':
        os.chdir('C:\\Users\\David\\OneDrive\\Projects\\nba-draft-player-analysis\\sources\\defense_regression_model\\steals')
        Xtrain,Xtest,ytrain,ytest = tts(df,cat_fields=stl_cat_fields,cont_fields=stl_cont_fields,response=stl_response)
        with open('steals_cols.txt', 'w') as f:
            for item in Xtrain.columns:
                f.write("%s\n" % item)
        train_models(Xtrain,Xtest,ytrain,ytest,model_name='steals_reg')
    
    if train_blocks == 'yes':
        os.chdir('C:\\Users\\David\\OneDrive\\Projects\\nba-draft-player-analysis\\sources\\defense_regression_model\\blocks')
        Xtrain,Xtest,ytrain,ytest = tts(df,cat_fields=blk_cat_fields,cont_fields=blk_cont_fields,response=blk_response)
        with open('blocks_cols.txt', 'w') as f:
            for item in Xtrain.columns:
                f.write("%s\n" % item)
        train_models(Xtrain,Xtest,ytrain,ytest,model_name='blocks_reg')

if predict_steals == 'yes' or predict_blocks =='yes':
    #Load Data
    os.chdir('C:\\Users\\David\\OneDrive\\Projects\\nba-draft-player-analysis')
    df = pd.read_csv(predict_file)
    df = df.set_index('name')
    df['usg_per_40'] = df['two_pointers_per_40']+df['three_point_attempts_per_40']+df['free_throw_attempts_per_40']
    
    if predict_steals == 'yes':
        os.chdir('C:\\Users\\David\\OneDrive\\Projects\\nba-draft-player-analysis\\sources\\defense_regression_model\\steals')
        probs = model_predict_reg(df=df,cat_fields=stl_cat_fields,cont_fields=stl_cont_fields,scaler='scaler.sav',
                                  encoder='encoder.sav',model='steals_reg.sav')
        stl_results = probs.set_index(df.index)
    
    if predict_blocks == 'yes':
        os.chdir('C:\\Users\\David\\OneDrive\\Projects\\nba-draft-player-analysis\\sources\\defense_regression_model\\blocks')
        probs = model_predict_reg(df=df,cat_fields=blk_cat_fields,cont_fields=blk_cont_fields,scaler='scaler.sav',
                                  encoder='encoder.sav',model='blocks_reg.sav')
        blk_results = probs.set_index(df.index)
    
    if stl_results is not None:
        stl_results.rename(columns={0:'steals_per_40_predict'},inplace=True)
        df_results = stl_results[['steals_per_40','steals_per_40_predict']]
    
    if blk_results is not None:
        blk_results.rename(columns={0:'blocks_per_40_predict'},inplace=True)
        if stl_results is not None:
            df_results = pd.merge(df_results,blk_results[['blocks_per_40','height','blocks_per_40_predict']],
                                  how='inner',left_index=True,right_index=True)
        else:
            df_results = blk_results[['blocks_per_40','blocks_per_40_predict']]
    
    os.chdir('C:\\Users\\David\\OneDrive\\Projects\\nba-draft-player-analysis\\predictions')
    df_results.round(2).to_csv('results_2020_defense.csv')
            