from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import math

import datetime
import json
import sys
import time

import random

def create_fake_dataset_for_eval(cur_phase, start_phase = 0):
    
    for phase in range(start_phase, cur_phase + 1):
        print('phase:',phase)
        click_test = pd.read_csv('datalab/61843/underexpose_test_click-{}.csv'.format(phase), header=None,  names=['user_id', 'item_id', 'time'])
        
        user_set = list(set(click_test['user_id']))
        time_asc_click_test = click_test.groupby('user_id').apply(lambda x:x.sort_values('time',ascending = True))

        fake_underexpose_test_qtime_with_answer = time_asc_click_test[time_asc_click_test['user_id']==user_set[0]][-1:]
        for i in range(1,len(user_set)):
            fake_underexpose_test_qtime_with_answer = pd.concat([fake_underexpose_test_qtime_with_answer,
                                                         time_asc_click_test[time_asc_click_test['user_id']==user_set[i]][-1:]])

        fake_underexpose_test_click = time_asc_click_test[time_asc_click_test['user_id']==user_set[0]][:-1]
        for i in range(1,len(user_set)):
            fake_underexpose_test_click = pd.concat([fake_underexpose_test_click,
                                            time_asc_click_test[time_asc_click_test['user_id']==user_set[i]][:-1]])
        
        fake_underexpose_test_qtime_with_answer.to_csv('/home/tianchi/myspace/fake_underexpose_test_qtime_with_answer-{}.csv'.format(phase), index = False, header = None)
        fake_underexpose_test_click.to_csv('/home/tianchi/myspace/fake_underexpose_test_click-{}.csv'.format(phase), index = False, header = None)
        
def _create_answer_file_for_evaluation(cur_phase, answer_fname):
    train = 'datalab/61843/underexpose_train_click-%d.csv'
    test = '/home/tianchi/myspace/fake_underexpose_test_click-%d.csv'


    answer = '/home/tianchi/myspace/fake_underexpose_test_qtime_with_answer-%d.csv'

    item_deg = defaultdict(lambda: 0)
    with open(answer_fname, 'w') as fout:
        for phase_id in range(cur_phase + 1):
            with open(train % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1
            with open(test % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1
            with open(answer % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    assert user_id % 11 == phase_id
                    print(phase_id, user_id, item_id, item_deg[item_id],
                          sep=',', file=fout)
                    
# 评估指标
def evaluate_each_phase(predictions, answers):
    list_item_degress = []
    for user_id in answers:
        item_id, item_degree = answers[user_id]['item_id'], answers[user_id]['item_degree']
        list_item_degress.append(item_degree)
    list_item_degress.sort()
    median_item_degree = list_item_degress[len(list_item_degress) // 2]

    num_cases_full = 0.0
    ndcg_50_full = 0.0
    ndcg_50_half = 0.0
    num_cases_half = 0.0
    hitrate_50_full = 0.0
    hitrate_50_half = 0.0
    for user_id in predictions:
        item_id, item_degree = answers[user_id]['item_id'], answers[user_id]['item_degree']
        rank = 1
        while rank <= 50 and int(predictions[user_id][rank]) != int(item_id):
            rank += 1
        num_cases_full += 1.0
        if rank <= 50:
            ndcg_50_full += 1.0 / np.log2(rank + 2.0)
            hitrate_50_full += 1.0
        if item_degree <= median_item_degree:
            num_cases_half += 1.0
            if rank <= 50:
                ndcg_50_half += 1.0 / np.log2(rank + 2.0)
                hitrate_50_half += 1.0
    ndcg_50_full /= num_cases_full
    hitrate_50_full /= num_cases_full
    ndcg_50_half /= num_cases_half
    hitrate_50_half /= num_cases_half
    return np.array([ndcg_50_full, ndcg_50_half,
                     hitrate_50_full, hitrate_50_half], dtype=np.float32)

def cal_hitrate_and_ndcg(model_version, cur_phase):
    scores = {}
    final_score = dict(zip(['ndcg_50_full', 'ndcg_50_half','hitrate_50_full', 'hitrate_50_half'],[0]*4))
    for c in range(cur_phase + 1):
        print('phase:',c)
        predictions = pd.read_csv('/home/tianchi/myspace/predictions_{}-{}.csv'.format(model_version,c),header = None)
        predictions = predictions.set_index(0).T.to_dict()

        answers = pd.read_csv('/home/tianchi/myspace/debias_track_answer-{}.csv'.format(c), header = None, names = ['phase_id','user_id','item_id','item_degree'])
        answers = answers[['user_id','item_id','item_degree']].set_index('user_id').T.to_dict()
        
        score = evaluate_each_phase(predictions, answers)
        scores['phase'+str(c)] = score
        final_score['ndcg_50_full'] += score[0]
        final_score['ndcg_50_half'] += score[1]
        final_score['hitrate_50_full'] += score[2]
        final_score['hitrate_50_half'] += score[3]
    return scores,final_score