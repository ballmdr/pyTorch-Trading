from flask import Flask
from torch_model import resnet18
import torch.nn.functional as F
import torch
import time
import fxcmpy
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import skimage as sk
from skimage.transform import resize
register_matplotlib_converters()


app = Flask(__name__)
K = 3
device = torch.device("cpu")
model = resnet18(K)
model.to(device)
access_token = '353dff029879f4e57f304310c1ab4137e8ff384f'
server = 'demo'
error_log = 'error.log'
mylog_path = 'pytorch_log.log'
img_path = 'torch_tmp.png'
timeframe = 'm30'
n_prices = 16
windows = 6

def writeLog(msg):
    #print(msg)
    file = open(mylog_path, 'a')
    file.write('\n')
    file.write(msg)
    file.close()

def connect():
    con_success = False
    while not con_success:
        try:
            writeLog('Connecting...')
            c = fxcmpy.fxcmpy(access_token=access_token, server=server, log_level='error', log_file=error_log)
        except:
            writeLog('Connect FAILED!')
            time.sleep(5)
        else:
            writeLog('Connect Success')
            con_success = True
            return c

def predictSignal(x, symbol):
    global model

    with torch.no_grad():
        weight_file = '../model/torch_' + symbol + '_m30'
        writeLog('Load file: ' + weight_file)
        model.load_state_dict(torch.load(weight_file, map_location='cpu'))
        xt = np.transpose(x, (0, 3, 1, 2))
        xt_tensor = torch.tensor(xt, dtype=torch.float32, device=device)
        logits = model(xt_tensor)
        logits_pred = F.softmax(logits, dim=1)
        pred = logits_pred.cpu().detach().numpy().argmax(1)

    # if pred == 2:
    #     return True
    # elif pred == 0:
    #     return False
    # else:
    #     return None
    writeLog('Pred: ' + str(pred))
    return pred

def getImg(p):
    windows = 6
    p['close'] = np.round((p.bidclose + p.askclose) / 2, 5)
    p['ma5'] = np.round(p.close.rolling(5).mean(), 5)
    p['ma7'] = np.round(p.close.rolling(7).mean(), 5)
    p['ma10'] = np.round(p.close.rolling(10).mean(), 5)
    p.dropna(inplace=True)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(3,2)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    _df = p.iloc[0:windows]
    plt.plot(_df.ma5, color='red')
    plt.plot(_df.ma7, color='blue')
    plt.plot(_df.ma10, color='green')
    #plt.plot(p.close)
    plt.savefig(img_path)
    #plt.show()
    plt.close()

    x = plt.imread(img_path)
    new_img = resize(x, (100,150))
    x = np.array([new_img], np.float32) / 255
    return x


@app.route('/predict')
def predict():
    s = request.args.get('s')
    symbol2 = s
    s = s.split('/')
    symbol = s[0] + s[1]

    #return jsonify({'symbol': symbol, 'symbol2': symbol2})
    con = connect()
    df = con.get_candles(symbol2, period=timeframe, number=n_prices)
    x = getImg(df)
    pred = predictSignal(x, symbol)
    json_data = jsonify({ 'pred': int(pred[0])})
    return json_data


if __name__ == '__main__':
    app.run()
