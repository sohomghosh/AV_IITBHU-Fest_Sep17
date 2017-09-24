######################################################################################################################################
############################################## Importing Packages ###################################################################
######################################################################################################################################
import pandas as pd
import numpy as np
from collections import Counter
import gc
from sklearn import preprocessing, model_selection, metrics, ensemble
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import f1_score


######################################################################################################################################
############################################## Reading Data ###################################################################
######################################################################################################################################

prob_data=pd.read_csv("/home/sohom/Desktop/AV_IITBHU_Enigma_Sep17/problem_data.csv")
#problem_id level_type  points tags
train_sub=pd.read_csv("/home/sohom/Desktop/AV_IITBHU_Enigma_Sep17/train_submissions.csv")
#user_id problem_id  attempts_range
user_data=pd.read_csv("/home/sohom/Desktop/AV_IITBHU_Enigma_Sep17/user_data.csv")
#user_id  submission_count  problem_solved  contribution  country follower_count  last_online_time_seconds  max_rating   rating rank  registration_time_seconds
test_sub=pd.read_csv("/home/sohom/Desktop/AV_IITBHU_Enigma_Sep17/test_submissions_NeDLEvX.csv")
#ID    user_id problem_id

##### Objective
#Find attempts_range

######################################################################################################################################
############################################## Preparing Dataset ###################################################################
######################################################################################################################################


train_user=pd.merge(train_sub,user_data,on='user_id',how='left')
train_user_prob=pd.merge(train_user,prob_data,on='problem_id',how='left')

test_user=pd.merge(test_sub,user_data,on='user_id',how='left')
test_user_prob=pd.merge(test_user,prob_data,on='problem_id',how='left')

train_user_prob['ID']=train_user_prob['user_id']+"_"+train_user_prob['problem_id']
test_user_prob['attempts_range']=np.nan


train_test=train_user_prob.append(test_user_prob)

######################################################################################################################################
############################################## Feature Engineering ###################################################################
######################################################################################################################################

##New feature: How old user is from registration_time; last online time from present time; tags as features
##country, tags, rank spelling mistakes check
#sort(set(sorted([str(i) for i in list(train_test['country'])], key=str.lower)))
#sorted(set(sorted([str(i) for i in list(train_test['country'])], key=str.lower)))

#sorted(set([str(i) for i in list(train_test['country'])]))
#sorted(set([str(i) for i in list(train_test['rank'])]))
tag_list=list(sorted(set([j for i in list(train_test['tags']) for j in str(i).split(',')])))
#['*special', '2-sat', 'binary search', 'bitmasks', 'brute force', 'chinese remainder theorem', 'combinatorics', 'constructive algorithms', 'data structures', 'dfs and similar', 'divide and conquer', 'dp', 'dsu', 'expression parsing', 'fft', 'flows', 'games', 'geometry', 'graph matchings', 'graphs', 'greedy', 'hashing', 'implementation', 'math', 'matrices', 'meet-in-the-middle', 'nan', 'number theory', 'probabilities', 'schedules', 'shortest paths', 'sortings', 'string suffix structures', 'strings', 'ternary search', 'trees', 'two pointers']


for tg in tag_list:
	train_test[tg]=train_test['tags'].apply(lambda x:1 if tg in str(x) else 0)



del train_test['tags']	
gc.collect()

tm=1506093368 #Present time
train_test['user_age_sec']=tm-train_test['registration_time_seconds']
train_test['last_online_sec']=tm-train_test['last_online_time_seconds']


del train_test['registration_time_seconds']
del train_test['last_online_time_seconds']

features=list(set(train_test.columns)-set(['ID','user_id','problem_id','attempts_range']))

print([(train_test[i].dtype,i) for i in features if train_test[i].dtype=='object'])
#[(train_test[i].dtype,i) for i in features]

'''
train_test['level_type'].value_counts()
A    85329
B    57870
C    37040
D    19667
E     8852
F     3326
G     2141
H     1667
J     1430
I     1382
K     1089
L      682
M      376
N       94
'''
train_test['level_type']=train_test['level_type'].replace(to_replace={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14})


country_list=list(set(train_test['country'])-set([np.nan]))
#[nan, 'Italy', 'Argentina', 'Azerbaijan', 'Jordan', 'Swaziland', 'Japan', 'Hungary', 'Christmas Island', 'Bulgaria', 'Romania', 'United Kingdom', 'Israel', 'Georgia', 'Mexico', 'Serbia', 'Costa Rica', 'Switzerland', 'Turkmenistan', 'North Korea', 'Peru', 'Poland', 'Tajikistan', 'Malaysia', 'Tunisia', 'Armenia', 'Bangladesh', 'Egypt', 'Taiwan', 'Iran', 'France', 'Indonesia', 'China', 'Australia', 'Netherlands', 'Moldova', 'Colombia', 'Macedonia', 'Germany', 'Cuba', 'Kyrgyzstan', 'Singapore', 'Bolivia', 'Spain', 'Morocco', 'Belarus', 'Russia', 'Mongolia', 'Laos', 'Brazil', 'South Korea', 'Venezuela', 'Iceland', 'Thailand', 'Uzbekistan', 'Czechia', 'Canada', 'Bosnia and Herzegovina', 'Slovakia', 'Lithuania', 'Haiti', 'Trinidad and Tobago', 'Norway', 'Croatia', 'Latvia', 'Philippines', 'South Africa', 'Vietnam', 'India', 'Kazakhstan', 'United States', 'Syria', 'Hong Kong', 'Chile', 'Finland', 'Estonia', 'Austria', 'Ukraine', 'Lebanon', 'Belgium']
for country_name in country_list:
	train_test[country_name]=train_test['country'].apply(lambda x :1 if str(x).lower().strip()==country_name.lower().strip() else 0)



del train_test['country']
gc.collect()

'''
train_test['rank'].value_counts()
intermediate    95037
beginner        76112
advanced        43184
expert           7517
'''
train_test['rank']=train_test['rank'].replace(to_replace={'beginner':1,'intermediate':2,'advanced':3,'expert':4})
train_test['unsolved']=train_test['submission_count']-train_test['problem_solved']
train_test['submission_count_diff']=max(train_test['submission_count'])-train_test['submission_count']

features=list(set(features+country_list+['unsolved','submission_count_diff'])-set(['country']))

'''
for f in [i for i in features if train_test[i].dtype=='object']:#Add all categorical features in the list
    lbl = LabelEncoder()
    lbl.fit(list(train_test[f].values))
    train_test[f] = lbl.transform(list(train_test[f].values))
'''

######################################################################################################################################
############################################## Building Model ###################################################################
######################################################################################################################################


params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1, "num_class": len(set(train_user_prob['attempts_range'])),
                "seed": 2016, "tree_method": "exact"}

X_train_all=train_test[0:len(train_user_prob.index)]
X_train_all['attempts_range']=X_train_all['attempts_range']-1
X_train_all['attempts_range']=pd.to_numeric(pd.Series(X_train_all['attempts_range']),errors='coerce')

X_train=X_train_all.sample(frac=0.80, replace=False)
X_valid=pd.concat([X_train_all, X_train]).drop_duplicates(keep=False)
X_test=train_test[len(train_user_prob.index):len(train_test.index)]


dtrain = xgb.DMatrix(X_train[features], X_train['attempts_range'], missing=np.nan)
dvalid = xgb.DMatrix(X_valid[features], X_train['attempts_range'], missing=np.nan)
dtest = xgb.DMatrix(X_test[features], missing=np.nan)

nrounds = 10000
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
valid_preds = bst.predict(dvalid)
valid_preds=[int(i)+1 for i in valid_preds]
test_preds = bst.predict(dtest)
test_preds=[int(i)+1 for i in test_preds]

print(f1_score(valid_preds,X_valid['attempts_range']+1,average='weighted'))

submit = pd.DataFrame({'ID': test_sub['ID'], 'attempts_range': test_preds})
submit[['ID','attempts_range']].to_csv("XGB8.csv", index=False)


### TO DO
#done#1) Make diffrent features : More Feature Engineering
	#'level_type': do own numeric encoding as per rank [avoid default], 'country': one-hot encode,'rank': do own numeric encoding as per rank [avoid default]
	#max(submission_count) - submission_count
	#submission_count - problem_solved

#2) Tune XGB parameters; also tune nrounds
#3) Make XGB with diffrent seeds and ensemble; Remove validation by doing X_train=X_train_all ; & then try 
#done#6) Ensemble all models created



#/After/ 4) Use catboost for categorical variables: Try properly
#/After After/ 5) Also try h2o, light gbm



################################## USING CATBOOST ######################################################
from catboost import CatBoostClassifier
cat_cols =[features.index(i) for i in ['level_type', 'country','rank']]
cols_to_use = features
model = CatBoostClassifier(depth=10, iterations=1000, learning_rate=0.1, eval_metric='F1', random_seed=1,loss_function='MultiClass',use_best_model=True)

model.fit(X_train[features],X_train['attempts_range'],cat_features=cat_cols,eval_set = (X_valid[features], X_valid['attempts_range']),use_best_model = True)
pred_valid = model.predict(X_valid[features])
pred_valid_ans=[int(i)+1 for i in list(pred_valid[:,0])]
print(f1_score(pred_valid_ans,X_valid['attempts_range']+1,average='weighted'))
pred = model.predict(X_test[features])
pred_ans=[int(i)+1 for i in list(pred[:,0])]

submit_cat = pd.DataFrame({'ID': test_sub['ID'], 'attempts_range': pred_ans})
submit_cat[['ID','attempts_range']].to_csv("catboost2.csv", index=False)


