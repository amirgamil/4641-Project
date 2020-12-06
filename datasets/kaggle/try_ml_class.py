import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tensorflow import keras
import sqlite3
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

cnx = sqlite3.connect("FPA_FOD_20170508.sqlite")
df = pd.read_sql_query("SELECT * FROM Fires LIMIT 20000", cnx)


df = df[['FIRE_YEAR', 'DISCOVERY_DATE', 'DISCOVERY_DOY', 'DISCOVERY_TIME',
    'STAT_CAUSE_CODE', 'STAT_CAUSE_DESCR', 'CONT_DATE', 'CONT_DOY', 'CONT_TIME',
    'FIRE_SIZE', 'FIRE_SIZE_CLASS', 'LATITUDE', 'LONGITUDE', 'STATE', 'COUNTY']]

df['X_COORD'] = np.cos(df['LATITUDE']) * np.cos(df['LONGITUDE'])
df['Y_COORD'] = np.cos(df['LATITUDE']) * np.sin(df['LONGITUDE'])
df['Z_COORD'] = np.sin(df['LATITUDE'])

df = df.drop(['LATITUDE', 'LONGITUDE'], axis=1)

s = df['DISCOVERY_TIME'].fillna('0000').str
df['DISCOVERY_TIMESTAMP'] = pd.to_datetime(df['FIRE_YEAR'], format="%Y") + \
    pd.to_timedelta(df['DISCOVERY_DOY'], unit='d') + \
    pd.to_timedelta(s[0:2].astype(int), unit='h') + \
    pd.to_timedelta(s[2:4].astype(int), unit='m')
df['DISCOVERY_TIME'] = s[0:2].astype(int) * 60 + s[2:4].astype(int)

s = df['CONT_TIME'].fillna('0000').str

df['CONTAINED_TIMESTAMP'] = pd.to_datetime(df['FIRE_YEAR'], format="%Y") + \
    pd.to_timedelta(df['CONT_DOY'], unit='d') + \
    pd.to_timedelta(s[0:2].astype(int), unit='h') + \
    pd.to_timedelta(s[2:4].astype(int), unit='m')
df['CONT_TIME'] = s[0:2].astype(int) * 60 + s[2:4].astype(int)

df['FIRE_DURATION'] = df['CONTAINED_TIMESTAMP'] - df['DISCOVERY_TIMESTAMP']

df = df.drop(['DISCOVERY_DATE', 'CONT_DATE', 'FIRE_YEAR'], axis=1)

df['STAT_CAUSE_CODE'] = df['STAT_CAUSE_CODE'].astype(int)
df['CONT_DOY'] = df['CONT_DOY'].fillna(df['DISCOVERY_DOY'] + 3).astype(int)

df['COUNTY'] = df['COUNTY'].fillna(-1).astype(int)

df['STAT_CAUSE_CODE'] = df['STAT_CAUSE_CODE'] - 1

uniques = df['FIRE_SIZE_CLASS'].unique()
numOuts = len(uniques)
uniqueMap = {v: i for i, v in enumerate(uniques)}
df['FIRE_SIZE_CLASS'] = df['FIRE_SIZE_CLASS'].apply(lambda x: uniqueMap[x])

other = df['STATE'].unique()
otherNum = len(other)
otheruniqueMap = {v: i for i, v in enumerate(other)}
df['STATE'] = df['STATE'].apply(lambda x: otheruniqueMap[x])

toSample = max([df[df['FIRE_SIZE_CLASS'] == i].shape[0] for i in range(numOuts)])
dd = df.groupby('FIRE_SIZE_CLASS').sample(toSample, replace=True)

targets = dd['FIRE_SIZE_CLASS']

# target_names = ['Lightning', 'Equipment Use', 'Smoking', 'Campfire', 'Debris Burning',
#                 'Railroad', 'Arson', 'Children', 'Miscellaneous', 'Fireworks', 'Powerline'
#                 'Structure', 'Missing/Undefined']
# print(targets.values)

# use only features which would be realistically accessible
columns = ['DISCOVERY_DOY', 'DISCOVERY_TIME', 'STAT_CAUSE_CODE', 'COUNTY', 'STATE', 'X_COORD', 'Y_COORD', 'Z_COORD']
dataset = dd[columns]

std_scale = StandardScaler().fit(dataset[['DISCOVERY_DOY', 'DISCOVERY_TIME', 'STAT_CAUSE_CODE', 'X_COORD', 'Y_COORD', 'Z_COORD']])
dataset[['DISCOVERY_DOY', 'DISCOVERY_TIME', 'STAT_CAUSE_CODE', 'X_COORD', 'Y_COORD', 'Z_COORD']] = std_scale.transform(dataset[['DISCOVERY_DOY', 'DISCOVERY_TIME', 'STAT_CAUSE_CODE', 'X_COORD', 'Y_COORD', 'Z_COORD']])

checkpoint_dir = "tmp_cat/checkpoint"
size = len(dataset)
batch_size = 16
test_size = int(0.2 * size) # 5000
val_size = int(0.3 * size) # 10000
epochs = 2

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, save_weights_only=True, save_freq=5*batch_size)

tb_callback = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, embeddings_freq=0, update_freq='epoch')

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, mode='auto', restore_best_weights=True)

ds = tf.data.Dataset.from_tensor_slices((dataset, targets.values))

ds = ds.shuffle(size)

train = ds.skip(test_size + val_size).batch(batch_size)
test = ds.take(test_size).batch(batch_size)
val = ds.skip(test_size).take(val_size).batch(batch_size)


model = keras.Sequential([
    keras.layers.Dense(24, input_shape=(len(dataset.columns),), activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(numOuts, activation='softmax')
])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

history = model.fit(train, epochs=epochs, callbacks=[cp_callback], validation_data=val, )
            

results = model.evaluate(test)
print("test loss, test acc: ", results)