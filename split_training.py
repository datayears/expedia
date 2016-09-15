# this codes splits the training data into smaller subset to work with

import numpy as np # linear algebra
import pandas as pd 
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from patsy import dmatrices

def gen_smaller_val(df):
	#test does not have index
	X_val, X_test = train_test_split(df, test_size=0.1, random_state=42)
	print len(X_test)
	print len(df)
	X_train, X_test = train_test_split(X_val, test_size=0.01, random_state=42)
	X_test.to_csv('val01.csv', sep=',',index=True)
	X_train, X_test = train_test_split(X_val, test_size=0.001, random_state=42)
	X_test.to_csv('val001.csv', sep=',',index=True)


def gen_smaller_train(df):
	#train does not have index
	X_train, X_test = train_test_split(df, test_size=0.001, random_state=21)

	print len(X_test)
	print len(df)

	X_test.to_csv('train001.csv', sep=',',index=False)

	X_train, X_test = train_test_split(df, test_size=0.0001, random_state=21)
	X_test.to_csv('train0001.csv', sep=',',index=False)

	X_train, X_test = train_test_split(df, test_size=0.00001, random_state=21)
	X_test.to_csv('train00001.csv', sep=',',index=False)

	X_train, X_test = train_test_split(df, test_size=0.01, random_state=21)
	X_test.to_csv('train01.csv', sep=',',index=False)

	X_train, X_test = train_test_split(df, test_size=0.1, random_state=21)
	X_test.to_csv('train1.csv', sep=',',index=False)


df = pd.read_csv('Data/train.csv', skipinitialspace=True)
# train = pd.read_csv('../Data/test.csv',
#                     dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32},
#                     usecols=['srch_destination_id','is_booking','hotel_cluster'],
#                     chunksize=1000000)

gen_smaller_val(df)
gen_smaller_train(df)

# aggs = []
# print '-'*38
# for chunk in train:
#     agg = chunk.groupby(['srch_destination_id',
#                          'hotel_cluster'])['is_booking'].agg(['sum','count'])
#     agg.reset_index(inplace=True)
#     aggs.append(agg)
#     # print '.',end=''
# print ''
# aggs = pd.concat(aggs, axis=0)
# aggs.head()