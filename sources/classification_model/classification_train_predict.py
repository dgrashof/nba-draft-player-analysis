import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

train = 'no'
predict = 'yes'

train_file = 'data/modeling_data_classification.csv'
predict_file = 'data/predict_2017.csv'


os.chdir('C:\\Users\\David\\OneDrive')
from utils.ml_data_proc import data_proc_class,model_predict
os.chdir('C:\\Users\\David\\OneDrive\\Projects\\nba-draft-player-analysis')

cont_fields = ['assists_per_40', 'blocks_per_40','free_throw_attempts_per_40','personal_fouls_per_40', 
               'points_per_40','steals_per_40', 'three_point_attempts_per_40','two_point_percentage',
               'three_point_percentage','free_throw_percentage','total_rebounds_per_40', 'turnovers_per_40',
               'two_point_attempts_per_40','height']

cat_fields = ['yrs_played']

response = 'class'

cols = ['yrs_played_1','yrs_played_4',
'free_throw_attempts_per_40',
'points_per_40', 
'yrs_played_2', 
'yrs_played_3',
'total_rebounds_per_40', 
'blocks_per_40', 
'assists_per_40', 
'steals_per_40',
'two_point_percentage',
'two_point_attempts_per_40', 
'three_point_attempts_per_40', 
'personal_fouls_per_40', 
'height',
'three_point_percentage', 
'free_throw_percentage', 
'turnovers_per_40', 
'yrs_played_5'] 



if train == 'yes':
    #load training data
    df = pd.read_csv(train_file)
    df = df.set_index('name')
    #identify categorical and continuous fields
    
    #process data - train, test, split/one hot encoding/imputing/feature selection
    Xtrain,ytrain,Xtest,ytest,cols = data_proc_class(df=df,cat_fields = cat_fields,cont_fields=cont_fields,response=response)
    
    with open('sources\cols.txt', 'w') as f:
        for item in cols:
            f.write("%s\n" % item)
    
    # Train Models
    # =============================================================================
    # 1.1 | Logistic Regression
    # =============================================================================

    model1 = LogisticRegression(random_state=42)

    param_grid = [{
        "fit_intercept" : [True, False],
        "warm_start"    : [True, False],
        "penalty"       : ['l2'],
        "solver"        : ['newton-cg','lbfgs'],
        "C"             : [1/10000,1/1000,1/100,1/10,1,10,100,1000,10000]
    }]

    # Evaluate performance of optimal model on test data.
    grid_search = GridSearchCV(model1, param_grid, cv=5, scoring="accuracy", verbose=0,n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)
    bestmodel1 = grid_search.best_estimator_

    # Apply bestmodel1 to out-of-sample test data `Xtest`.
    model1_pred = bestmodel1.predict(Xtest)

    # Asses performance of model5.
    print("Logistic Regression Results")
    print("\n")
    print(metrics.classification_report(ytest, model1_pred))
    
    # =============================================================================
    # 1.2 | Random Forest Classifier
    # =============================================================================

    model2 = RandomForestClassifier(random_state=42)

    param_grid = [{
        "n_estimators": [10, 25, 50, 75, 100, 150, 200, 250, 500],
        "max_depth": [None, 2, 5, 10, 15],
        "criterion": ["gini", "entropy"],
        "bootstrap": [True, False],
        "warm_start": [True, False],
    }]

    # Evaluate performance of optimal model on test data.
    grid_search = GridSearchCV(model2, param_grid, cv=5, scoring="accuracy", verbose=0,n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)
    bestmodel2 = grid_search.best_estimator_

    # Apply bestmodel1 to out-of-sample test data `Xtest`.
    model2_pred = bestmodel2.predict(Xtest)

    # Asses performance of model1.
    print("Random Forest Results")
    print("\n")
    print(metrics.classification_report(ytest, model2_pred))
    

    # RandomForestClassifier can produce feature importances.
    importances = bestmodel2.feature_importances_
    feature_indx = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature Importances")
    plt.bar(Xtest.columns[feature_indx], importances[feature_indx], color="r",
            align="center")
    plt.xticks(rotation=90)
    plt.xlim([-1, Xtrain.shape[1]])
    plt.show()
    
    # =============================================================================
    # 1.3 | XGB Classifier
    # =============================================================================
    
    params = {
        "n_estimators": [10, 25, 50, 75, 100, 150, 200, 250, 500],
        "max_depth": [None, 2, 5, 10, 15],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]}
    
    model3 = XGBClassifier(learning_rate=0.02, objective='multi:softprob',random_state=42)
    
    # Evaluate performance of optimal model on test data.
    grid_search = GridSearchCV(model3, param_grid, cv=5, scoring="accuracy", verbose=0,n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)
    bestmodel3 = grid_search.best_estimator_
    
    # Apply bestmodel1 to out-of-sample test data `Xtest`.
    model3_pred = bestmodel3.predict(Xtest)

    # Asses performance of model1.
    print("XGB Classifier Results")
    print("\n")
    print(metrics.classification_report(ytest, model3_pred))
    

    # RandomForestClassifier can produce feature importances.
    importances = bestmodel3.feature_importances_
    feature_indx = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature Importances")
    plt.bar(Xtest.columns[feature_indx], importances[feature_indx], color="r",
            align="center")
    plt.xticks(rotation=90)
    plt.xlim([-1, Xtrain.shape[1]])
    plt.show()
    
    # =============================================================================
    # 1.4 | Stacking Model Classifier
    # =============================================================================
    
    #embed models in Classifier and use Logistic Regression as Final Estimator
    stacking_model = StackingClassifier(
    estimators=[("log",bestmodel1),("rf",bestmodel2),("xgb",bestmodel3)],final_estimator = LogisticRegression())
    stacking_model.fit(Xtrain,ytrain)
    stacking_model_predict = stacking_model.predict(Xtest)
    stacking_model_predict_proba = stacking_model.predict_proba(Xtest)
    print("Stacking Classifier Results")
    print("\n")
    print(metrics.classification_report(ytest, stacking_model_predict))
    
    # Serialize model to load for future predictions/avoid re-training
    pickle.dump(stacking_model, open('sources\model_class.sav', 'wb'))


if predict == 'yes':
    #Load Data
    df = pd.read_csv(predict_file)
    df = df.set_index('name')

    #trans_cols = ['assists','blocks','free_throw_attempts','personal_fouls','points','steals','three_point_attempts','total_rebounds',
    #        'turnovers','two_point_attempts']

    #for i in trans_cols:
    #    df[i+'_per_40'] = round((df[i]/df['minutes_played'])*40,2)

    #df.drop(trans_cols,axis = 1,inplace = True)
    
    probs = model_predict(df=df,cat_fields=cat_fields,cont_fields=cont_fields,scaler='sources\scaler.sav',encoder='sources\encoder.sav',
              cols=cols,model='sources\model_class.sav')
    
    results = probs.set_index(df.index)
    results.round(2).sort_values(by='all_star',ascending=False).to_csv('results_2017.csv')