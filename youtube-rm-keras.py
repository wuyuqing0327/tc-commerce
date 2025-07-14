import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
# from tensorflow.keras import initializers
# from tensorflow.keras.callbacks import LearningRateScheduler
# from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import json
from tqdm import tqdm
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score

# 1。 avg(item1 + item2 + item3 + item4) 与 每一个item——embeeding 去比较，取前200个
# 2。 精排 xgboost：一个用户对应200个物品，有哪些点击了，点击为1， 不点击为0
# user_id1, item_embedding1, 0
# user_id1, item_embedding2, 1
# user_id1, item_embedding3, 0
# user_id2, item_embedding1, 1
# user_id2, item_embedding2, 1
# user_id2, item_embedding3, 0


# 一步到位：deep fm/wide&deep

embedding_size = 128
item_num = 50
learn_rate = 1e-4
top_k = 50
max_item_len = 4
stride = 1
max_user_len = 19
model_path = 'model/'

item_text_dict = {}
with open('data/vocab_txt.json', "r", encoding='utf-8') as reader:
    json_config1 = json.loads(reader.read())
for key, value in json_config1.items():
    item_text_dict[key] = value

item_img_dict = {}
with open('data/vocab_txt.json', "r", encoding='utf-8') as reader:
    json_config2 = json.loads(reader.read())
for key, value in json_config2.items():
    item_img_dict[key] = value

userdf = pd.read_csv('data/underexpose_train/underexpose_user_feat.csv', header=None)
userdf.columns = ['user_id', 'user_age_level', 'user_gender', 'user_city_level']

userdf['user_gender_M'] = np.where(userdf['user_gender'] == 'M', 1, 0)
userdf['user_gender_F'] = np.where(userdf['user_gender'] == 'F', 1, 0)
userdf['user_gender_Missing'] = np.where(pd.isnull(userdf['user_gender']), 1, 0)

userdf['user_age_level_1'] = np.where(userdf['user_age_level'] == 1, 1, 0)
userdf['user_age_level_2'] = np.where(userdf['user_age_level'] == 2, 1, 0)
userdf['user_age_level_3'] = np.where(userdf['user_age_level'] == 3, 1, 0)
userdf['user_age_level_4'] = np.where(userdf['user_age_level'] == 4, 1, 0)
userdf['user_age_level_5'] = np.where(userdf['user_age_level'] == 5, 1, 0)
userdf['user_age_level_6'] = np.where(userdf['user_age_level'] == 6, 1, 0)
userdf['user_age_level_7'] = np.where(userdf['user_age_level'] == 7, 1, 0)
userdf['user_age_level_8'] = np.where(userdf['user_age_level'] == 8, 1, 0)
userdf['user_age_level_Missing'] = np.where(pd.isnull(userdf['user_age_level']), 1, 0)

userdf['user_city_level_1'] = np.where(userdf['user_city_level'] == 1, 1, 0)
userdf['user_city_level_2'] = np.where(userdf['user_city_level'] == 2, 1, 0)
userdf['user_city_level_3'] = np.where(userdf['user_city_level'] == 3, 1, 0)
userdf['user_city_level_4'] = np.where(userdf['user_city_level'] == 4, 1, 0)
userdf['user_city_level_5'] = np.where(userdf['user_city_level'] == 5, 1, 0)
userdf['user_city_level_6'] = np.where(userdf['user_city_level'] == 6, 1, 0)
userdf['user_city_level_Missing'] = np.where(pd.isnull(userdf['user_city_level']), 1, 0)

usercol = ['user_gender_M', 'user_gender_F', 'user_gender_Missing',
           'user_age_level_1', 'user_age_level_2', 'user_age_level_3',
           'user_age_level_4', 'user_age_level_5', 'user_age_level_6',
           'user_age_level_7', 'user_age_level_8', 'user_age_level_Missing',
           'user_city_level_1', 'user_city_level_2', 'user_city_level_3',
           'user_city_level_4', 'user_city_level_5', 'user_city_level_6',
           'user_city_level_Missing']
colname = usercol.copy()
colname.append('user_id')
userdf = userdf[colname]
user_dict = {}
a = list(userdf['user_id'])
for i in range(userdf.shape[0]):
    user_dict[str(a[i])] = list(userdf.iloc[i, :-1])

# ---------------------------------------------------------------------------------------------------
train_path = 'data/train.csv'
test_path = 'data/test.csv'


def data_transformer(path, user_dict, max_item_len, stride):
    """
    :csv_input:  1,  78142,  0.9837416195438412 (columns=['user_id', 'item_id', 'time'])
    :param path: read data path
    :param max_item_len: windows of observation item
    :param stride: stride
    :return:
    [('1', ['78142', '26646', '89568', '76240'], '87533'),
     ('1', ['76240', '87533', '78380', '85492'], '97795')]
    """
    if stride > max_item_len:
        print("strip is over max_item_len for observation item windows")
        return []
    itemid = []
    sample = []
    lastuserid = 1
    setuserid = []
    setitemid = []
    with open(path, 'r') as f:
        for line in tqdm(f):
            a = line.split(',')
            setuserid.append(int(a[0]))
            setitemid.append(int(a[1]))
            user_var = user_dict.get(a[0], [0] * 19)
            userid = int(a[0])
            if userid == lastuserid:
                itemid.append(int(a[1]))
                if len(itemid) > max_item_len:
                    next_item_id = itemid[-1]
                    next_re_input_id = itemid[stride:]
                    sample.append((userid, itemid[:max_item_len], user_var, next_item_id))
                    itemid = []
                    itemid.extend(next_re_input_id)
                lastuserid = int(a[0])
            else:
                if max_item_len >= len(itemid) > (max_item_len - stride + 1):
                    if lastuserid != sample[-1][0]:
                        itemid = []
                    else:
                        next_item_id = itemid[-1]
                        need_pad_num = max_item_len - len(itemid[:-1])
                        last_item_list = sample[-1][1]
                        idx = max_item_len - stride
                        need_item = last_item_list[(-idx - need_pad_num):-idx]
                        need_item.extend(itemid[:-1])
                        sample.append((lastuserid, need_item, user_var, next_item_id))
                        itemid = [int(a[1])]
                else:
                    itemid = [int(a[1])]
                lastuserid = int(a[0])
        f.close()
    return sample, set(setuserid), set(setitemid)


train_data, train_userid, train_itemid = data_transformer(train_path, user_dict, max_item_len=max_item_len,
                                                          stride=stride)
test_data, test_userid, test_itemid = data_transformer(test_path, user_dict, max_item_len=max_item_len, stride=stride)

item_index = list(set(item_text_dict.keys()) | set(train_itemid))
item_txt_embedding_matrix = np.zeros((117644, embedding_size))
item_img_embedding_matrix = np.zeros((117644, embedding_size))
for i in tqdm(range(len(item_index))):
    word = item_index[i]
    embedding_vector1 = item_text_dict.get(word)
    embedding_vector2 = item_img_dict.get(word)
    if embedding_vector1 is not None and embedding_vector2 is not None:
        # words not found in embedding index will be all-zeros.
        item_txt_embedding_matrix[int(word)] = embedding_vector1
        item_img_embedding_matrix[int(word)] = embedding_vector2


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def generate(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            _X1, _X2, _Y = [], [], []
            for i in idxs:
                d = self.data[i]
                x1 = d[1]
                x2 = d[2]
                y = int(d[-1])
                y1 = item_txt_embedding_matrix[y]
                y2 = item_img_embedding_matrix[y]
                _X1.append(x1)
                _X2.append(x2)
                _X_array1 = np.array(_X1)
                _X_array2 = np.array(_X2)
                _Y.append(np.concatenate((y1, y2)))
                _Y_array = np.array(_Y)

                if len(_X1) == self.batch_size or i == idxs[-1]:
                    _X1, _X2, _Y = [], [], []
                    yield (_X_array1, _X_array2), _Y_array


train_D = data_generator(train_data)
test_D = data_generator(test_data)

# ---- model -------------------------------------------------------------------------------------
vist_item_Input = Input(shape=(max_item_len,), name='vist_item_Input', dtype='float32')
user_Input = Input(shape=(max_user_len,), name='user_Input', dtype='float32')

visit_items_txt_embedding = Embedding(input_dim=item_txt_embedding_matrix.shape[0], output_dim=embedding_size,
                                      input_length=max_item_len, weights=[item_txt_embedding_matrix],
                                      trainable=False,
                                      name="item_txt_embedding")(vist_item_Input)
visit_items_txt_average_embedding = GlobalAveragePooling1D()(visit_items_txt_embedding)
visit_items_img_embedding = Embedding(input_dim=item_txt_embedding_matrix.shape[0], output_dim=embedding_size,
                                      input_length=max_item_len, weights=[item_img_embedding_matrix],
                                      trainable=False,
                                      name="item_img_embedding")(vist_item_Input)
visit_items_img_average_embedding = GlobalAveragePooling1D()(visit_items_img_embedding)

input_embedding = Concatenate(name='input_embedding')(
    [visit_items_txt_average_embedding, visit_items_img_average_embedding, user_Input])
# kernel_initializer_1 = tf.initializers.random_normal(mean=0.0, stddev=0.1)
# bias_initializer_1 = tf.initializers.random_normal(mean=0.0, stddev=0.1)
layer_1 = Dense(128, activation='relu', name="layer_1")(input_embedding)
# layer_dropout_1 = Dropout(0.2, name="layer_dropout_1")(layer_1)

# kernel_initializer_2 = tf.initializers.random_normal(mean=0.0, stddev=0.1)
# bias_initializer_2 = tf.initializers.random_normal(mean=0.0, stddev=0.1)
layer_2 = Dense(64, activation='relu', name="layer_2")(layer_1)
# layer_dropout_2 = Dropout(0.2, name="layer_dropout_2")(layer_2)

# kernel_initializer_3 = tf.initializers.random_normal(mean=0.0, stddev=0.1)
# bias_initializer_3 = tf.initializers.random_normal(mean=0.0, stddev=0.1)
user_vector = Dense(embedding_size * 2, activation='relu', name="user_vector")(layer_2)
model = Model(inputs=(vist_item_Input, user_Input), outputs=user_vector)

# user_vector 与每一个item_embedding 求余弦相似度，取前200个，看能召回多少。
def my_cross_entropy(target_embedding, user_vector):
    logits = tf.matmul(user_vector, target_embedding, transpose_a=False, transpose_b=True)  # num * num
    yhat = tf.nn.softmax(logits)
    cross_entropy = tf.reduce_mean(-tf.log(tf.matrix_diag_part(yhat) + 1e-16))
    return cross_entropy


opt = keras.optimizers.SGD(learning_rate=learn_rate, decay=1e-6, momentum=0.9, nesterov=True,
                           name='SGD')
# opt.minimize(my_cross_entropy, var_list=[next_visit_item_Input, user_vector])

model.compile(loss=my_cross_entropy,
              # optimizer=opt,
              # optimizer = Lookahead(best_optimizer=best_optimizer, k=5, alpha=0.5),
              optimizer=keras.optimizers.Adam(learning_rate=learn_rate),
              # optimizer=LazyOptimizer(keras.optimizers.Adam(), ["embedding"]),
              metrics=[my_cross_entropy])
# lookahead = Lookahead(k=5, alpha=0.5)  # 初始化Lookahead
# lookahead.inject(model)  # 插入到模型中
model.summary()

path = '/Users/yuqingwu/Workspace/tc-ecommerce/model/weights.epoch-18--tr_loss-2.6762-val_loss-2.7336.h5'
model = load_model(path, custom_objects={'my_cross_entropy': my_cross_entropy})

# ------------ model_train -----------------------------------------------------------------------------------------
# class RocAucMetricCallback(keras.callbacks.Callback):
#     def __init__(self, traination_data, validation_data, include_on_batch=False, validation=True):
#         super(RocAucMetricCallback, self).__init__()
#         self.a = traination_data
#         self.x = None
#         self.y = None
#         self.x_val = validation_data[0]
#         self.y_val = validation_data[1]
#         self.include_on_batch = include_on_batch
#         self.validation = validation
#         self.batch_auc = []
#         self.batch = None
#         self.epoch_count = 0
#         self.batch_count = 0
#
#     def on_train_begin(self, logs={}):
#         if not ('train_roc_auc' in self.params['metrics']):
#             self.params['metrics'].append('train_roc_auc')
#         if not ('val_roc_auc' in self.params['metrics']):
#             self.params['metrics'].append('val_roc_auc')
#
#     def on_train_end(self, logs={}):
#         pass
#
#     def on_batch_begin(self, batch, logs={}):
#         if (self.include_on_batch):
#             train = next(self.a)
#             self.x = train[0]
#             self.y = train[1]
#             self.batch_count += 1
#         pass
#
#     def on_batch_end(self, batch, logs={}):
#         if (self.include_on_batch):
#             auc = roc_acc(self.y, self.model.predict(self.x))
#             self.batch_auc.append(auc)
#
#     def on_epoch_begin(self, epoch, logs={}):
#         self.epoch_count += 1
#         pass
#
#     def on_epoch_end(self, epoch, logs={}):
#         if (self.include_on_batch):
#             train_auc = sum(self.batch_auc) / self.batch_count
#             logs['train_auc'] = train_auc
#         if (self.validation):
#             # self.model.save(model_path + 'epoch-%d.h5' %self.epoch_count)
#             self.x = self.a[0]
#             self.y = self.a[1]
#             logs['train_roc_auc'] = roc_acc(self.y, self.model.predict(self.x))
#             logs['val_roc_auc'] = roc_acc(self.y_val, self.model.predict(self.x_val))


model.fit_generator(
    train_D.generate(),
    steps_per_epoch=train_D.steps,
    epochs=30,
    validation_data=test_D.generate(),
    validation_steps=test_D.steps,
    callbacks=[
        # RocAucMetricCallback((train_X_array, train_Y_array), (valid_X_array, valid_Y_array)),
        keras.callbacks.ModelCheckpoint(
            model_path + "weights.epoch-{epoch:02d}--tr_loss-{my_cross_entropy:.4f}-val_loss-{val_my_cross_entropy:.4f}.h5",
            monitor='val_my_cross_entropy',
            mode='min', save_best_only=False, verbose=1),
        # predict_test,
        # keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, verbose=1, mode='max'),
        # TensorBoard(log_dir=model_path + 'logs', write_graph=True, write_images=True,
        #             histogram_freq=1, update_freq='batch', write_grads=True)
    ]
)
# 35+30+30 epoch

# ------------------- predict ------------------------------------------------------------------

def my_predict(valid, top_k):
    rank = []
    for user in tqdm(range(100)):
        userid = valid[user][0]
        user_vector_predict = model.predict((np.array([valid[user][1]]), np.array([valid[user][2]])))
        target = valid[user][3]
        item_rating = {}
        for i in range(len(item_txt_embedding_matrix)):
            y1 = item_txt_embedding_matrix[i]
            y2 = item_img_embedding_matrix[i]
            item_embedding = np.concatenate((y1, y2))
            logits = np.dot(user_vector_predict, item_embedding.reshape(-1, 1))  # num * num
            yhat = np.exp(-logits)/(1+np.exp(-logits))
            item_rating[i] = yhat[0][0]
        ratings = sorted(item_rating.items(), key=lambda item: item[1])
        top_item = ratings[:top_k]
        ratings = dict(ratings)
        rank_value = sum(np.array(list(ratings.values())) <= ratings[target])
        rank.append([userid, rank_value, top_item])
    return rank

train_rank = my_predict(train_data, top_k)
test_rank = my_predict(test_data, top_k)

np.array(train_rank)[:,1]

# ----------------------------------------------------------------------------------------------
idxs = list(range(10))
np.random.shuffle(idxs)
X1, X2, Y = [], [], []
for i in tqdm(idxs):
    d = train_data[i]
    x1 = d[1]
    x2 = d[2]
    y = d[-1]
    X1.append(x1)
    X2.append(x2)
    X1_array = np.array(X1)
    X2_array = np.array(X2)
    Y.append(y)
    Y_array = np.array(Y)
train_X_array, train_Y_array = (X1_array, X2_array), Y_array
#
idxs = list(range(len(test_data)))
np.random.shuffle(idxs)
X1, X2, Y = [], [], []
for i in tqdm(idxs):
    d = test_data[i]
    x1 = d[1]
    x2 = d[2]
    y = d[-1]
    X1.append(x1)
    X2.append(x2)
    X1_array = np.array(X1)
    X2_array = np.array(X2)
    Y.append(y)
    Y_array = np.array(Y)
test_X_array, test_Y_array = (X1_array, X1_array), Y_array
