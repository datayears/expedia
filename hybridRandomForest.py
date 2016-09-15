# coding: utf-8
# https://www.kaggle.com/maxcvc/expedia-hotel-recommendations/valium-1/run/242385/code 
#   0.50062

__author__ = 'Ravi: https://kaggle.com/company'

import datetime
from heapq import nlargest
from operator import itemgetter
import math
import numpy as np

def prepare_arrays_match():
    f = open("Data/train.csv", "r")
    f.readline()
    
    best_hotels_od_uid = dict()
    best_hotels_od_ulc = dict()
    best_hotels_uid_miss = dict()
    best_hotels_search_dest = dict()
    best_hotels_country = dict()
    popular_hotel_cluster = dict()
    best_s00 = dict()
    best_s01 = dict()
    cpt6 = dict()

    total = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 2000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        if arr[11] != '':
            book_year = int(arr[11][:4])
            book_month = int(arr[11][5:7])
        else:
            book_year = int(arr[0][:4])
            book_month = int(arr[0][5:7])
            
        if book_month<1 or book_month>12 or book_year<2012 or book_year>2015:
            #print(book_month)
            #print(book_year)
            #print(line)
            continue
            
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        user_id = arr[7]
        is_package = arr[9]
        srch_destination_id = arr[16]
        hotel_country = arr[21]
        hotel_market = arr[22]
        is_booking = float(arr[18])
        hotel_cluster = arr[23]

        append_0 = ((book_year - 2012)*12 + (book_month - 12))
        if not (append_0>0 and append_0<=36):
            # print(book_year)
            # print(book_month)
            # print(line)
            # print(append_0)
            continue
        
        append_1 = pow(math.log(append_0), 1.3) * pow(append_0, 1.45)* (3.5 + 17.60*is_booking)
        append_2 = 3 + 5.56*is_booking     

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '' and hotel_country != '' and is_booking==1:
            s00 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
            if s00 in best_s00:
                if hotel_cluster in best_s00[s00]:
                    best_s00[s00][hotel_cluster] += append_0
                else:
                    best_s00[s00][hotel_cluster] = append_0
            else:
                best_s00[s00] = dict()
                best_s00[s00][hotel_cluster] = append_0

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '' and is_booking==1:
            s01 = (user_id, srch_destination_id, hotel_country, hotel_market)
            if s01 in best_s01:
                if hotel_cluster in best_s01[s01]:
                    best_s01[s01][hotel_cluster] += append_0
                else:
                    best_s01[s01][hotel_cluster] = append_0
            else:
                best_s01[s01] = dict()
                best_s01[s01][hotel_cluster] = append_0


        if user_location_city != '' and orig_destination_distance == '' and user_id !='' and srch_destination_id != '' and hotel_country != '' and is_booking==1:
            s0 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
            if s0 in best_hotels_uid_miss:
                if hotel_cluster in best_hotels_uid_miss[s0]:
                    best_hotels_uid_miss[s0][hotel_cluster] += append_0
                else:
                    best_hotels_uid_miss[s0][hotel_cluster] = append_0
            else:
                best_hotels_uid_miss[s0] = dict()
                best_hotels_uid_miss[s0][hotel_cluster] = append_0

        if srch_destination_id != '' and orig_destination_distance != '' and is_booking==1:
            s1 = (srch_destination_id, orig_destination_distance)
            if s1 in best_hotels_od_uid:
                if hotel_cluster in best_hotels_od_uid[s1]:
                    best_hotels_od_uid[s1][hotel_cluster] += append_0
                else:
                    best_hotels_od_uid[s1][hotel_cluster] = append_0
            else:
                best_hotels_od_uid[s1] = dict()
                best_hotels_od_uid[s1][hotel_cluster] = append_0

        if hotel_market != '' and hotel_country != '':
            s1 = (hotel_market,hotel_country)
            if s1 in cpt6:
                if hotel_cluster in cpt6[s1]:
                    cpt6[s1][hotel_cluster] += append_1
                else:
                    cpt6[s1][hotel_cluster] = append_1
            else:
                cpt6[s1] = dict()
                cpt6[s1][hotel_cluster] = append_1

        if user_location_city != '' and orig_destination_distance != '':
            s1 = (user_location_city, orig_destination_distance)

            if s1 in best_hotels_od_ulc:
                if hotel_cluster in best_hotels_od_ulc[s1]:
                    best_hotels_od_ulc[s1][hotel_cluster] += append_0
                else:
                    best_hotels_od_ulc[s1][hotel_cluster] = append_0
            else:
                best_hotels_od_ulc[s1] = dict()
                best_hotels_od_ulc[s1][hotel_cluster] = append_0

        if srch_destination_id != '' and hotel_country != '' and hotel_market != '':
            s2 = (srch_destination_id,hotel_country,hotel_market,is_package)
            if s2 in best_hotels_search_dest:
                if hotel_cluster in best_hotels_search_dest[s2]:
                    best_hotels_search_dest[s2][hotel_cluster] += append_1
                else:
                    best_hotels_search_dest[s2][hotel_cluster] = append_1
            else:
                best_hotels_search_dest[s2] = dict()
                best_hotels_search_dest[s2][hotel_cluster] = append_1

        if hotel_market != '':
            s3 = (hotel_market)
            if s3 in best_hotels_country:
                if hotel_cluster in best_hotels_country[s3]:
                    best_hotels_country[s3][hotel_cluster] += append_2
                else:
                    best_hotels_country[s3][hotel_cluster] = append_2
            else:
                best_hotels_country[s3] = dict()
                best_hotels_country[s3][hotel_cluster] = append_2

        if hotel_cluster in popular_hotel_cluster:
            popular_hotel_cluster[hotel_cluster] += append_0
        else:
            popular_hotel_cluster[hotel_cluster] = append_0

    f.close()
    return best_hotels_od_uid, best_s00,best_s01, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster,cpt6


def ctreeConstruct():
    import time
    import pandas as pd 
    from sklearn.ensemble import RandomForestClassifier
    s=time.time()
    print "START"
    df = pd.read_csv('Data/train.csv', dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32},parse_dates=True, infer_datetime_format=True)
    e=time.time()
    print e-s

    df['date_time'] = pd.to_datetime(df.date_time,errors='coerce')
    df['book_year'] = df['date_time'].dt.year
    from sklearn import tree
    # X = df[['hotel_market', 'hotel_country','hotel_continent','srch_destination_id','user_location_city','user_id','book_year','user_location_country','user_location_region','is_package','srch_destination_type_id']]
    X = df[['hotel_market', 'hotel_country','hotel_continent','srch_destination_id','user_location_city','user_id','book_year','is_package']]
    target = df.hotel_cluster
    ctreeModel = RandomForestClassifier(n_estimators=5,min_samples_leaf=150)
    ctreeModel = ctreeModel.fit(X, target)

    s=time.time()
    print "START test"
    df_val = pd.read_csv('Data/test.csv', dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32},parse_dates=True, infer_datetime_format=True)
    e=time.time()
    print e-s
    df_val['date_time'] = pd.to_datetime(df_val.date_time,errors='coerce')
    df_val['book_year'] = df_val['date_time'].dt.year
    X_val = df_val[['hotel_market', 'hotel_country','hotel_continent','srch_destination_id','user_location_city','user_id','book_year','is_package']]
    probs = ctreeModel.predict_proba(X_val)
    print np.shape(probs)

    # from sklearn.externals import joblib
    # joblib.dump(ctreeModel, 'RandomeForestModel.pkl') 

    return probs

def gen_submission(probs,best_hotels_od_uid,best_s00, best_s01,best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster,cpt6):
    now = datetime.datetime.now()
    path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    f = open("Data/test.csv", "r")
    f.readline()
    total = 0
    total0 = 0
    total00 = 0
    total1 = 0
    total2 = 0
    total3 = 0
    total4 = 0
    total5 = 0 
    total6 = 0 
    out.write("id,hotel_cluster\n")
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 100000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        user_id = arr[8]
        is_package = arr[10]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]

        out.write(str(id) + ',')
        filled = []

        s1 = (user_location_city, orig_destination_distance)
        if s1 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s1]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total1 += 1

        # s1 = (srch_destination_id, orig_destination_distance)
        # if s1 in best_hotels_od_uid:
        #     d = best_hotels_od_uid[s1]
        #     topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
        #     for i in range(len(topitems)):
        #         if topitems[i][0] in filled:
        #             continue
        #         if len(filled) == 5:
        #             break
        #         out.write(' ' + topitems[i][0])
        #         filled.append(topitems[i][0])
        #         total5 += 1

        if orig_destination_distance == '':
            s0 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
            if s0 in best_hotels_uid_miss:
                d = best_hotels_uid_miss[s0]
                topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
                for i in range(len(topitems)):
                    if topitems[i][0] in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(topitems[i][0])
                    total0 += 1

        s00 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
        s01 = (user_id, srch_destination_id, hotel_country, hotel_market)
        if s01 in best_s01 and s00 not in best_s00:
            d = best_s01[s01]
            topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total00 += 1

        s2 = (srch_destination_id,hotel_country,hotel_market,is_package)
        if s2 in best_hotels_search_dest:
            d = best_hotels_search_dest[s2]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])
                total2 += 1

        # s2 = (hotel_market,hotel_country)
        # if s2 in cpt6:
        #     d = cpt6[s2]
        #     topitems = nlargest(5, d.items(), key=itemgetter(1))
        #     for i in range(len(topitems)):
        #         if topitems[i][0] in filled:
        #             continue
        #         if len(filled) == 5:
        #             break
        #         out.write(' ' + topitems[i][0])
        #         filled.append(topitems[i][0])
        #         total6+= 1

        # s3 = (hotel_market)
        # if s3 in best_hotels_country:
        #     d = best_hotels_country[s3]
        #     topitems = nlargest(5, d.items(), key=itemgetter(1))
        #     for i in range(len(topitems)):
        #         if topitems[i][0] in filled:
        #             continue
        #         if len(filled) == 5:
        #             break
        #         out.write(' ' + topitems[i][0])
        #         filled.append(topitems[i][0])
        #         total3 += 1

        ls = probs[int(id)]
        predict_top5 = nlargest(5, xrange(len(ls)), key=ls.__getitem__)
        topitems = map(str,predict_top5)
        for i in range(len(topitems)):
            if topitems[i] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' ' + topitems[i])
            filled.append(topitems[i])
            total5 += 1

        if len(filled)<5:
            print len(filled)
            print id
        # for i in range(len(topclasters)):
        #     if topclasters[i][0] in filled:
        #         continue
        #     if len(filled) == 5:
        #         break
        #     out.write(' ' + topclasters[i][0])
        #     filled.append(topclasters[i][0])
        #     total4 += 1

        out.write("\n")
    out.close()
    print('Total 1: {} ...'.format(total1))
    print('Total 0: {} ...'.format(total0))
    print('Total 00: {} ...'.format(total00))
    print('Total 2: {} ...'.format(total2))
    print('Total 3: {} ...'.format(total3))
    print('Total 4: {} ...'.format(total4))
    print('Total 5: {} ...'.format(total5))
    print('Total 6: {} ...'.format(total6))
    print total 
    print id


probs=ctreeConstruct()
best_hotels_od_uid,best_s00,best_s01,best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster,cpt6 = prepare_arrays_match()
gen_submission(probs,best_hotels_od_uid,best_s00, best_s01,best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster,cpt6)