import pandas as pd
import numpy as np
import re
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import fasttext

train=pd.read_csv("/home/sohom/Desktop/AV_IITBHU_Lingupedia_Sep17/train_E6oV3lV.csv")
test=pd.read_csv("/home/sohom/Desktop/AV_IITBHU_Lingupedia_Sep17/test_tweets_anuFYb8.csv")

test['label']=np.nan
train_test=train.append(test)

sentences_split=[re.split('\W', i) for i in train_test['tweet']]

#wordvec
model_w2v = word2vec.Word2Vec(sentences_split, size=40,min_count =1, window=3, workers =-1,sample=1e-5)
features_sent = np.zeros(shape=(0,40))
for i in sentences_split:
	su=np.zeros(shape=(40))
	for j in i:
		k=np.array(model_w2v.wv[j])
		su=su+k
		#print(su)
	features_sent=np.vstack([features_sent, su])


#tfidf
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=.01, use_idf=True, smooth_idf=False, sublinear_tf=True)
sklearn_representation = sklearn_tfidf.fit(train_test['tweet'])
train_test_tfidf=pd.DataFrame(sklearn_tfidf.transform(train_test['tweet']).todense())


#fasttext
train_test['tweet'].to_csv('train_test_tweet.csv',index=False)
model_sk = fasttext.skipgram('train_test_tweet.csv', 'model_sk',dim=40)
features_sent_ft = np.zeros(shape=(0,40))
for i in sentences_split:
	su=np.zeros(shape=(40))
	for j in i:
		k=np.array(model_sk[j])
		su=su+k
		#print(su)
	features_sent_ft=np.vstack([features_sent_ft, su])



#Keeping w2v, fasttext and tfidf side by side
train_test_features=pd.concat([pd.DataFrame(features_sent),pd.DataFrame(features_sent_ft),train_test_tfidf],axis=1)

X_train=train_test_features[0:len(train.index)]
X_test=train_test_features[len(train.index):len(train_test_features.index)]
X_train.columns=["feature_"+str(i) for i in range(0,X_train.shape[1])]
X_test.columns=["feature_"+str(i) for i in range(0,X_test.shape[1])]

dtrain = xgb.DMatrix(X_train, train['label'], missing=np.nan)
dtest = xgb.DMatrix(X_test, missing=np.nan)

nrounds = 1000
watchlist = [(dtrain, 'train')]
params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4,"num_class": 2, "silent": 1,"eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,"min_child_weight": 1,"seed": 2016, "tree_method": "exact"}
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
test_preds = bst.predict(dtest)

submit = pd.DataFrame({'id': test['id'], 'label': test_preds})
submit[['id','label']].to_csv("xgb3.csv", index=False)

###Others to do
#0) clean text
#1) Use smote to balance
#2) Make handcrafter features corresponding to hashtags