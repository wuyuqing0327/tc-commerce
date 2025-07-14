import pandas as pd
import numpy as np
from tqdm import tqdm

userdf = pd.read_csv('data/underexpose_train/underexpose_user_feat.csv', header=None)
userdf.columns = ['user_id', 'user_age_level', 'user_gender', 'user_city_level']
userdf['user_gender'] = np.where(userdf['user_gender']=='M', 1,
                                 np.where(userdf['user_gender']=='F', 0, 2))
userdf['user_age_level'] = np.where(pd.isnull(userdf['user_age_level']), 5.0, userdf['user_age_level'])
userdf['user_city_level'] = np.where(pd.isnull(userdf['user_city_level']), 4.0, userdf['user_city_level'])

user_dict = {}
for i in range(userdf.shape[0]):
    user_dict[list(userdf['user_id'])[i]] = list(userdf.iloc[i, 1:])

import json_tricks
with open('data/user_dict.json', 'w') as f:
    f.write(json_tricks.dumps(user_dict, primitives=True))


# itemdf = pd.read_csv('data/underexpose_train/underexpose_item_feat.csv', header=None)
item_text_dict = {}
item_image_dict = {}
with open('data/underexpose_train/underexpose_item_feat.csv', 'r') as f:
    for line in tqdm(f):
        a = line.strip('\n').replace('[', '').replace(']', '').split(',')
        item_text_dict[a[0]] = np.array(a[1:129]).astype(np.float32)
        item_image_dict[a[0]] = np.array(a[129:]).astype(np.float32)


import json_tricks
import json
with open('data/vocab_txt.json', 'w') as f:
    f.write(json_tricks.dumps(item_text_dict, primitives=True))
    f.close()
with open('data/vocab_img.json', 'w') as f:
    f.write(json_tricks.dumps(item_image_dict, primitives=True))
    f.close()

item_dict = {}
with open('data/vocab.json', "r", encoding='utf-8') as reader:
    json_config = json.loads(reader.read())
for key, value in json_config.items():
    item_dict[key] = value


user_dict = {}
with open('data/user_dict.json', "r", encoding='utf-8') as reader:
    json_config = json.loads(reader.read())
for key, value in json_config.items():
    user_dict[key] = value
# ----------------------------------------------------------------------------------------------

train_click_0 = pd.read_csv('data/underexpose_train/underexpose_train_click-0.csv', header=None)
train_click_1 = pd.read_csv('data/underexpose_train/underexpose_train_click-1.csv', header=None)
train_click_2 = pd.read_csv('data/underexpose_train/underexpose_train_click-2.csv', header=None)
train_click_3 = pd.read_csv('data/underexpose_train/underexpose_train_click-3.csv', header=None)
train_click_4 = pd.read_csv('data/underexpose_train/underexpose_train_click-4.csv', header=None)



train_click_0.columns = ['user_id', 'item_id', 'time']
train_click_1.columns = ['user_id', 'item_id', 'time']
train_click_2.columns = ['user_id', 'item_id', 'time']
train_click_3.columns = ['user_id', 'item_id', 'time']
train_click_4.columns = ['user_id', 'item_id', 'time']



train = pd.concat([train_click_0, train_click_1, train_click_2,train_click_3,train_click_4], axis=0)
train = train.sort_values(['user_id', 'time'])
train = train.drop_duplicates()
train = train.reset_index(drop=True)


# train_data = train.groupby(['user_id'], as_index=False)['item_id'].count()
# train_data['item_id'] = train.groupby(['user_id'], as_index=False).apply(lambda x : list(x['item_id']))



'''
lstm:{user_id1: itemid1, itemid2, ...... itemid5, label: itemid6,
      user_id1: itemid2, itemid3, ...... itemid6, label: itemid7}

bert:{user_id: itemid1, itmeid2, ...... itemidn,   label: itemidmask}
'''

test_click_0 = pd.read_csv('data/underexpose_test/underexpose_test_click-0/underexpose_test_click-0.csv', header=None)
test_qtime_0 = pd.read_csv('data/underexpose_test/underexpose_test_click-0/underexpose_test_qtime-0.csv', header=None)
test_click_1 = pd.read_csv('data/underexpose_test/underexpose_test_click-1/underexpose_test_click-1.csv', header=None)
test_qtime_1 = pd.read_csv('data/underexpose_test/underexpose_test_click-1/underexpose_test_qtime-1.csv', header=None)
test_click_2 = pd.read_csv('data/underexpose_test/underexpose_test_click-2/underexpose_test_click-2.csv', header=None)
test_qtime_2 = pd.read_csv('data/underexpose_test/underexpose_test_click-2/underexpose_test_qtime-2.csv', header=None)
test_click_3 = pd.read_csv('data/underexpose_test/underexpose_test_click-3/underexpose_test_click-3.csv', header=None)
test_qtime_3 = pd.read_csv('data/underexpose_test/underexpose_test_click-3/underexpose_test_qtime-3.csv', header=None)
test_click_4 = pd.read_csv('data/underexpose_test/underexpose_test_click-4/underexpose_test_click-4.csv', header=None)
test_qtime_4 = pd.read_csv('data/underexpose_test/underexpose_test_click-4/underexpose_test_qtime-4.csv', header=None)


test_click_0.columns = ['user_id', 'item_id', 'time']
test_qtime_0.columns = ['user_id', 'query_time']
test_click_1.columns = ['user_id', 'item_id', 'time']
test_qtime_1.columns = ['user_id', 'query_time']
test_click_2.columns = ['user_id', 'item_id', 'time']
test_qtime_2.columns = ['user_id', 'query_time']
test_click_3.columns = ['user_id', 'item_id', 'time']
test_qtime_3.columns = ['user_id', 'query_time']
test_click_4.columns = ['user_id', 'item_id', 'time']
test_qtime_4.columns = ['user_id', 'query_time']



test_click = pd.concat([ test_click_0, test_click_1,test_click_2,test_click_3,test_click_4], axis=0)
test_qtime = pd.concat([test_qtime_0, test_qtime_1,test_qtime_2,test_qtime_3,test_qtime_4], axis=0)


test = test_click.sort_values(['user_id', 'time'])
test = test.drop_duplicates()
test = test.reset_index(drop=True)


print(train.shape, test.shape, test_qtime.shape)
print(train.columns, test.columns)

train.to_csv('data/train.csv', index=False, header=None)
test.to_csv('data/test.csv', index=False, header=None)



