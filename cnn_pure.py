#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
import fxcmpy
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, concatenate, Flatten, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
#%%
def connect():
    return fxcmpy.fxcmpy(access_token=access_token, server=server)

def norm(x):
    return x
    #return (x-x.mean())/x.std()
#%%
access_token = '757cf3bec951c194aaa61c9db954c6533d251f2f'
server = 'real'
symbol = 'BTCUSD'
symbol2 = 'BTC/USD'
timeframe = 'm30'
n_prices = 10000
windows = 6
target_windows = 3
K = 3

#%%
con = connect()

#%%
df = con.get_candles(symbol2, period=timeframe, number=n_prices, columns=['bidclose', 'askclose'])

df['close'] = (df.askclose + df.bidclose) / 2
df['ma5'] = df.close.rolling(5).mean()
df['ma10'] = df.close.rolling(10).mean()
df['ma20'] = df.close.rolling(20).mean()

df.close = norm(df.close)
df.ma5 = norm(df.ma5)
df.ma10 = norm(df.ma10)
df.ma20 = norm(df.ma20)
df.dropna(inplace=True)
drop_cols = ['askclose', 'bidclose']
df.drop(drop_cols, axis=1, inplace=True)

#%%

X = list()
Y = list()
for i in range(len(df)-windows-target_windows+1):
    end = i + windows
    X.append(df.iloc[i:end,:].values)
    Y.append(df.iloc[i+windows+target_windows-1,0])

X = np.array(X)
Y = np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)



#%%
cnn1 = Sequential()
cnn1.add(Conv1D(filters=16, kernel_size=2, use_bias=False, strides=2, padding='same', kernel_initializer='he_normal', input_shape=(windows, X.shape[2])))
cnn1.add(BatchNormalization())
cnn1.add(Activation('relu'))
cnn1.add(Conv1D(filters=32, kernel_size=2, use_bias=False, strides=2, padding='same', kernel_initializer='he_normal'))
cnn1.add(BatchNormalization())
cnn1.add(Activation('relu'))
cnn1.add(MaxPooling1D(pool_size=2))
cnn1.add(Flatten())

cnn2 = Sequential()

cnn2.add(Conv1D(filters=16, kernel_size=2, use_bias=False, strides=2, padding='same', kernel_initializer='he_normal', input_shape=(windows, X.shape[2])))
cnn2.add(BatchNormalization())
cnn2.add(Activation('relu'))
cnn2.add(Conv1D(filters=32, kernel_size=2, use_bias=False, strides=2, padding='same', kernel_initializer='he_normal'))
cnn2.add(BatchNormalization())
cnn2.add(Activation('relu'))
cnn2.add(MaxPooling1D(pool_size=2))
cnn2.add(Flatten())

model.add(Dense(50, use_bias=False, kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))

early_stop = EarlyStopping(patience=30, monitor='val_loss', mode='min')
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

#%%
model.fit(X, Y, epochs=1000, verbose=2, validation_data=(x_test, y_test), callbacks=[early_stop])

#%%
index = np.random.randint(0, len(x_test)-1)
yHat = model.predict(x_test[index].reshape(1, windows, X.shape[2]))
print(yHat)
print(y_test[index])

#%%
