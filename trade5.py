import torch
from Flask_server.torch_model import resnet18
import torch.nn.functional as F
import time
import signal
from datetime import datetime
from time import gmtime
import fxcmpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import skimage as sk
from skimage.transform import resize
register_matplotlib_converters()

def handler(signum, frame):
    print('handler')
    raise Exception('Action took too much time')

def getNewPrice():
    global price, con
    if not con.is_connected:
        con = connect()
    writeLog('get price for ' + symbol2)
    # update pricedata on first attempt
    have_new_price = False
    c = 0
    while not have_new_price:
        new_price = con.get_candles(symbol2, period=timeframe, number=n_prices)
        if len(new_price) > 0:
            have_new_price = True
        else:
            c += 1
            time.sleep(3)
            if c > 3:
                con = connect()

    if len(price[symbol] == 0):
        price[symbol] = new_price
        return True


    if new_price.index.values[-1] != price[symbol].index.values[-1]:
        price[symbol] = new_price
        return True

    counter = 0
    # If data is not available on first attempt, try up to 3 times to update pricedata
    while new_price.index.values[-1] == price[symbol].index.values[-1] and counter < 10:
        writeLog("No updated prices found, trying again in 10 seconds...")
        time.sleep(10)
        if counter == 5:
            con = connect()
        new_price = con.get_candles(symbol2, period=timeframe, number=n_prices)
        counter += 1
    if new_price.index.values[-1] != price[symbol].index.values[-1]:
        price[symbol] = new_price
        return True
    else:
        return False


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

def getDistance(x):
    mask = offer['currency'] == symbol2
    if len(offer[mask]) > 0:
        pip = offer[mask].pip.values[0]
        cost = offer[mask].pipCost.values[0]
    else:
        if symbol2.split('/')[1] != 'JPY':
            pip = 0.0001
            cost = 0.0001
        else:
            pip = 0.01
            cost = 0.00009

    distance = x * pip / cost
    if pip == 0.01:
        distance /= 10
    elif pip == 0.0001:
        distance *= 10
    else:
        writeLog('pip not valid!')
    return np.round(distance, 0)

def openPos(isBuy):
    global con
    writeLog('Open ' + str(isBuy))
    trades = con.get_open_trade_ids()

    if profit is not None:
        limit = getDistance(profit)
    else:
        limit = None

    if loss is not None:
        stop = - getDistance(loss)
    else:
        stop = None

    writeLog('limit: ' + str(limit))
    #print('stop: ', stop)

    signal.alarm(10)
    try:
        opentrade = con.open_trade(symbol=symbol2, is_buy=isBuy, is_in_pips=True, limit=limit, stop=stop, trailing_step=trailing_step, amount=amount, time_in_force='GTC', order_type='AtMarket')
    except:
        signal.alarm(0)
        writeLog('Open position not success')
        con = connect()
        return False
    else:
        signal.alarm(0)
        #print(opentrade)
        writeLog('Open ' + symbol2 + ' ' + str(isBuy) + ' Success')
        return True


def canOpen2():
    poses = con.get_open_trade_ids()
    print('check can open')
    print('len position: ', len(poses))
    #print('symbol2: ', symbol2)
    for trade_id in poses:
        pos = con.get_open_position(trade_id)
        currency = pos.get_currency()
        isBuy = pos.get_isBuy()
        stop = pos.get_stop()
        open_ = pos.get_open()
        print('currency: ', pos.get_currency())
        if symbol2 == currency:
            writeLog('Symbol Exists')
            if stop == 0:
                writeLog('Stop = 0, cannot open!')
                return False
            elif isBuy and stop < open_:
                writeLog('position buy, not breakeven yet!')
                return False
            elif not isBuy and stop > open_:
                writeLog('position sell, not breakeven yet!')
                return False
    writeLog('Can open')
    return True


def checkSignal():
    global con, price
    if not con.is_connected:
        con = connect()
    writeLog('checking signal...' + symbol2)
    price[symbol] = con.get_candles(symbol2, period=timeframe, number=n_prices)
    isBuy = predictSignal()
    if isBuy is not None:
        if canOpen2():
            open_success = False
            while not open_success:
                open_success = openPos(isBuy)
                time.sleep(1)

def init(open_new = False):
    global symbol, symbol2, weight_file, model
    #checkPos()
    for i in range(len(list1)):
        symbol = list1[i]
        symbol2 = list2[i]
        writeLog(symbol2)
        price[symbol] = con.get_candles(symbol2, period=timeframe, number=n_prices)
        if open_new:
            checkSignal()
        else:
            writeLog('Not open new position')
        writeLog('--------')

def Update(force_update=False):
    global symbol, symbol2, model, con
    con = connect()
    writeLog('awakening...')
    #checkPos()
    for i in range(len(list1)):
        symbol = list1[i]
        symbol2 = list2[i]
        if force_update or getNewPrice():
            writeLog('Trade ' + symbol2)
            writeLog(str(datetime.now()) + " Got new prices -> Predicted Signal...")
            checkSignal()
        writeLog('-----------')
    writeLog('sleeping...')

def closeAllPos(currency = None):
    global con
    if currency is None:
        signal.alarm(10)
        try:
            close_trade = con.close_all()
        except:
            signal.alarm(0)
            writeLog('Close all Failed!')
            time.sleep(5)
            con = connect()
            return False
        else:
            signal.alarm(0)
            writeLog('Close all Success')
            return True

    else:
        writeLog('Close all ' + currency)
        trades = con.get_open_trade_ids()
        if len(trades) == 0:
            writeLog('No open position')
        else:
            sum_gross = 0
            for i in trades:
                pos = con.get_open_position(i)
                if pos.get_currency() == currency:
                    close_success = False
                    while not close_success:
                        close_success = closeTrade(trade_id=i, amount=amount)
                        time.sleep(5)

def checkProfit():
    sum_gross = con.get_open_positions_summary().grossPL.values[0]
    print('Sum gross: ', sum_gross)
    if sum_gross >= max_profit:
        close_all_success = False
        while not close_all_success:
            close_all_success = closeAllPos()


def writeLog(msg):
    print(msg)
    file = open(mylog_path, 'a')
    file.write('\n')
    file.write(msg)
    file.close()

def predictSignal():
    global model

    with torch.no_grad():
        weight_file = 'model/torch_' + symbol + '_m30'
        writeLog('load file: ' + weight_file)
        model.load_state_dict(torch.load(weight_file, map_location='cpu'))
        x = getImg()
        xt = np.transpose(x, (0, 3, 1, 2))
        xt_tensor = torch.tensor(xt, dtype=torch.float32, device=device)
        logits = model(xt_tensor)
        logits_pred = F.softmax(logits, dim=1)
        pred = logits_pred.cpu().detach().numpy().argmax(1)

    #print('predicted: ', pred)
    writeLog('predicted: ' + str(pred))

    if pred == 2:
        return True
    elif pred == 0:
        return False
    else:
        return None


def getImg(df):
    p = df
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


if __name__ == "__main__":

    device = torch.device("cpu")

    access_token = '353dff029879f4e57f304310c1ab4137e8ff384f'
    server = 'demo'

    list1 = ['GBPUSD', 'USDJPY', 'EURCAD', 'NZDCAD', 'CADCHF']

    list2 = ['GBP/USD', 'USD/JPY', 'EUR/CAD', 'NZD/CAD', 'CAD/CHF']

    timeframe = 'm30'
    n_prices = 16
    windows = 6
    price = dict()
    mylog_path = 'pytorch_log5.log'
    error_log = 'error5.log'
    img_path = 'torch_tmp5.png'
    amount = 1
    profit = 3
    max_profit = 0
    loss = 1
    trailing_step = 1
    model = None
    symbol = ''
    symbol2 = ''
    weight_file = ''

    signal.signal(signal.SIGALRM, handler)

    con = connect()
    offer = con.get_offers(kind='dataframe')

    model = resnet18(3)
    model.to(device)
    writeLog(str(datetime.now()))

    init(open_new=True)
    writeLog('Init finish')
    writeLog('-----------')

    while True:
        currenttime = datetime.now()
        # if currenttime.second == 0 and currenttime.minute % 5 == 0:
        #     checkProfit()
        #     time.sleep(240)
        if currenttime.second == 0 and currenttime.minute % 30 == 0:
            #checkProfit()
            Update()
            time.sleep(1500)
        time.sleep(1)
