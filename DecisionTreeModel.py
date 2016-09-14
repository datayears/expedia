import pandas as pd 
from patsy import dmatrices
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import tree
import heapq
import time
from average_precision import mapk,apk

s=time.time()
print "START"
df = pd.read_csv('Data/train001.csv', dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32},parse_dates=True, infer_datetime_format=True)
e=time.time()
print e-s

# convert to string to datetime
df['date_time'] = pd.to_datetime(df.date_time,errors='coerce')
df['srch_ci'] = pd.to_datetime(df.srch_ci,errors='coerce')
df['srch_co'] = pd.to_datetime(df.srch_co,errors='coerce')

print df.columns[:]
df = df.drop([u'orig_destination_distance'], axis=1)
#remove rows with missing value
df = df.dropna(axis=0)
print 'Number of entries: ' + str(len(df))
#convert date time to int 
df['num_day_stay'] = df['srch_co']-df['srch_ci'] 
df['num_day_stay']=(df['num_day_stay']/np.timedelta64(1, 'D')).astype(int)
df['weekday'] = df['srch_ci'].dt.dayofweek
# df['is_same_country'] = (df.user_location_country == df.hotel_country).astype(int)
# df['is_same_continent'] = (df.posa_continent == df.hotel_continent).astype(int)

print df.columns[:]
# df_reduce=df.drop(df.columns[[0,1,8,12,13,18,21,24]], axis=1)
df_reduce=df.drop(df.columns[[0,1,11,12,17,22,23]], axis=1)
y = df.hotel_cluster

# separate into train and test
X_train, X_test, y_train, y_test = train_test_split(df_reduce, y, test_size=0.4, random_state=10) 

model = tree.DecisionTreeClassifier(random_state=0,min_samples_leaf=50)
model = model.fit(X_train, y_train)

# examine the result 
print "the training data accuracy is: %.6f " % model.score(X_train, y_train)
print "the testing data accuracy is: %.6f " % model.score(X_test, y_test)

# predict class labels for the test set 
# predicted = model.predict(X_test) 
# print predicted

# generate class probabilities and find the top 5 highest class
probs = model.predict_proba(X_test) #matrix
y_test_ls = y_test.as_matrix()
# predict_top5 = []
map5score = []

for i in range(len(probs)):
    ls = probs[i]
    # predict_top5[i] = heapq.nlargest(5, xrange(len(ls)), key=ls.__getitem__)
    predict_top5 = heapq.nlargest(5, xrange(len(ls)), key=ls.__getitem__)
    actual = []
    actual.append(y_test_ls[i])
    score = apk(actual,predict_top5,5)
    map5score.append(score)

# map5score = mapk(y_test_ls,predict_top5,5)

# print count/len(probs)
print "the testing data MAP5 is: %f " % np.mean(map5score)

# analysis the feature 
ls_feat_val= model.feature_importances_
ls_feat_name = list(X_train.columns.values)

ls_index = heapq.nlargest(10, xrange(len(ls_feat_val)), key=ls_feat_val.__getitem__)
ls_feat_name_top = list(ls_feat_name[i] for i in ls_index)
print ls_feat_name_top


# tree.export_graphviz(model,out_file='tree.dot')
# $dot -Tpng tree.dot -o tree.png
