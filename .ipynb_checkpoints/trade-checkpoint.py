#!/home/ballmdr/anaconda3/bin/python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
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


def predictSignal():
    x = getImg()
    xt = np.transpose(x, (0, 3, 1, 2))
    xt_tensor = torch.tensor(xt, dtype=torch.float32, device=device)
    logits = model(xt_tensor)
    logits_pred = F.softmax(logits, dim=1)
    pred = logits_pred.cpu().detach().numpy().argmax(1)

    if test_pred:
        pred = np.random.randint(0,3)

    #print('predicted: ', pred)
    writeLog('predicted: ' + str(pred))

    if pred == 2:
        return True
    elif pred == 0:
        return False
    else:
        return None


def getImg():
    p = price[symbol]
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
    plt.savefig('torch_tmp.png')
    #plt.show()
    plt.close()

    x = plt.imread('torch_tmp.png')
    new_img = resize(x, (100,150))
    x = np.array([new_img], np.float32) / 255
    return x

def writeLog(msg):
    print(msg)
    file = open(mylog_path, 'a')
    file.write('\n')
    file.write(msg)
    file.close()

def connect():
    try:
        c = fxcmpy.fxcmpy(access_token=access_token, server=server, log_level='error', log_file='error2.log')
    except:
        c = None
    else:
        return c

def checkPos():
    global con
    return 0
    if not con.is_connected:
        con = connect()
    print('Check Position...')
    trades = con.get_open_trade_ids()
    if len(trades) == 0:
        print('No open position')
    else:
        for i in trades:
            pos = con.get_open_position(i)
            pos_time = pos.get_time()
            #print('fxcm time: ', pos_time)
            pos_amount = pos.get_amount()
            #print(pos.get_currency())
            c_old = pos_time.strftime('%H:%M')
            #print(c_old)
            c_old = c_old.split(':')
            c_new = time.strftime("%H:%M", gmtime())
            #print(c_new)
            c_new = c_new.split(':')

            c_new[0] = int(c_new[0])
            c_new[1] = int(c_new[1])
            c_old[0] = int(c_old[0])
            c_old[1] = int(c_old[1])

            if c_new[0] < c_old[0]:
                c_new[0] += 24

            if c_new[1] >= 30:
                c_new[1] = 30
            else:
                c_new[1] = 0

            if c_old[1] >= 30:
                c_old[1] = 30
            else:
                c_old[1] = 0

            origin = (c_old[0]*60) + c_old[1]
            current = (c_new[0]*60) + c_new[1]

            delta = np.abs(current - origin)
            #print('Current time: ', current)
            #print('Time open: ', origin)
            #print('Delta: ', delta)
            if delta >= delta_close:
                print('Close time exceed!')
                close_success = False
                while not close_success:
                    if closeTrade(i, pos_amount):
                        close_success = True
                    else:
                        con = connect()
            else:
                print('Not exceed')

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

def closeTrade(trade_id, amount):
    global con
    try:
        close_trade = con.close_trade(trade_id=trade_id, amount=amount)
    except:
        writeLog('Close failed Retry...')
        con = connect()
        return False
    else:
        writeLog('Close success')
        return True

def getSumGross(currency):
    trades = con.get_open_trade_ids()
    if len(trades) == 0:
        writeLog('No open position')
        return False
    else:
        sum_gross = 0
        for i in trades:
            pos = con.get_open_position(i)
            if pos.get_currency() == currency:
                sum_gross += pos.get_grossPL()
        return sum_gross

def closeAllPos(currency = None):
    if currency is None:
        con.close_all()
        return 0
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
                    time.sleep(1)

def canOpen():
    writeLog('Check can open?')
    sum_gross = getSumGross(symbol2)
    writeLog('sum gross: ' + str(sum_gross))
    if not sum_gross:
        return True
    if sum_gross < loss_ratio:
        return True
    elif sum_gross >= win_ratio:
        closeAllPos(symbol2)
        return True
    else:
        writeLog('Cannot open!')
        return False

def checkSignal(force=True):
    global con, price

    if not con.is_connected:
        con = connect()
    writeLog('checking signal...' + symbol2)
    price[symbol] = con.get_candles(symbol2, period=timeframe, number=n_prices)
    isBuy = predictSignal()
    if isBuy is not None:
        if force or canOpen():
            open_success = False
            while not open_success:
                open_success = openPos(isBuy)
                time.sleep(1)

def canOpenLego():
    trades = con.get_open_trade_ids()
    if len(trades) == 0:
        writeLog('No position, can open')
        return True
    else:
        pos = con.get_open_position(trades[-1])
        gross = pos.get_grossPL()
        writeLog('Gross last position ' + str(gross))
        if gross < loss_ratio or pos.get_stop() != 0:
            writeLog('Find that pos is exists')
            for i in trades:
                pos = con.get_open_position(i)
                if symbol2 == pos.get_currency():
                    writeLog(symbol2 + ' exist!')
                    return False
            writeLog('can open')
            return True
        else:
            writeLog('not yet loss, cannot open ')
            return False

def init(open_new = False):
    global symbol, symbol2, weight_file, model
    #checkPos()
    for i in range(len(list1)):
        symbol = list1[i]
        symbol2 = list2[i]
        writeLog(symbol2)
        price[symbol] = con.get_candles(symbol2, period=timeframe, number=n_prices)
        if open_new:
            weight_file = 'torch_' + symbol + '_m30'
            model.load_state_dict(torch.load(weight_file, map_location='cpu'))
            model.eval()
            checkSignal()
        else:
            writeLog('Not open new position')

        writeLog('--------')

def Update(force_update=False):
    global symbol, symbol2, model, con
    if not con.is_connected:
        con = connect()
    writeLog('awakening...')
    #checkPos()
    for i in range(len(list1)):
        symbol = list1[i]
        symbol2 = list2[i]
        if force_update or getNewPrice():
            writeLog('Trade ' + symbol2)
            writeLog(str(datetime.now()) + " Got new prices -> Predicted Signal...")
            weight_file = 'torch_' + symbol + '_m30'
            writeLog('load file: ' + weight_file)
            model.load_state_dict(torch.load(weight_file, map_location='cpu'))
            model.eval()
            checkSignal()
        writeLog('-----------')
    writeLog('sleeping...')

def checkLego():
    writeLog('Check Lego')
    trades = con.get_open_trade_ids()
    trades.sort()
    if len(trades) != 0:
        pos = con.get_open_position(trades[-1])
        gross1 = pos.get_grossPL()
    writeLog('Len position ' + str(len(trades)))
    try:
        summary = con.get_open_positions_summary().grossPL.values[0]
    except:
        summary = 0
    writeLog('All Summary ' + str(summary))
    if gross1 < loss_ratio or pos.get_stop() != 0:
        # loss -> open new layer
        writeLog('Loss! open new poition')
        Update(force_update=True)
    else:
        if len(trades) == 1:
            if gross1 >= win_ratio:
                writeLog('move stop and set limit')
                con.change_trade_stop_limit(trades[-1], is_in_pips=False, is_stop=True, rate=pos.get_open())
                con.change_trade_stop_limit(trades[-1], is_in_pips=True, is_stop=False, rate=getDistance(const_profit))
                writeLog('open new position')
                Update(force_update=True)
        else: # len trades > 1
            if summary >= 0:
                closeAllPos()
                Update(force_update=True)
            else:
                pos = con.get_open_position(trades[-2])
                gross2 = pos.get_grossPL()
                summary = gross1 + gross2
                writeLog('Sum of 2 last ' + str(summary))
                if summary >= 0:
                    writeLog('Close 2 lastest')
                    close_success = False
                    while not close_success:
                        close_success = closeTrade(trades[-1], amount)
                    close_success = False
                    while not close_success:
                        close_success = closeTrade(trades[-2], amount)
                    trades = con.get_open_trade_ids()
                    if len(trades) == 0:
                        Update(force_update=True)



# In[11]:


##########################
### MODEL
##########################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 16
        if grayscale:
            in_dim = 1
        else:
            in_dim = 4
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 16, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear((256*2*3) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = F.dropout(x)

        x = self.layer1(x)
        x = F.dropout(x)
        x = self.layer2(x)
        x = F.dropout(x)
        x = self.layer3(x)
        x = F.dropout(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = F.dropout(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        x = F.dropout(x)

        return logits


def resnet18(num_classes):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   num_classes=num_classes,
                   grayscale=False)
    return model


# In[15]:


if __name__ == "__main__":
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
    device = torch.device("cpu")

    access_token = '82e41ea2bcf020754f67aeb835e9ad617dbc5427'
    server = 'demo'
    # new : euraud, eurchf, eurnzd, gbpaud, gbpnzd, cadjpy
    # new : audusd, nzdusd
    # agressive : eurusd, usdchf, eurcad, eurgbp, gbpcad, gbpjpy, audcad, audnzd, chfjpy
    # conservative : gbpusd, audchf, audjpy
    # t-ded : usdjpy, usdcad, eurjpy, gbpchf, nzdcad, nzdchf, cadchf

    list1 = ['USDJPY', 'USDCAD', 'EURJPY', 'GBPCHF', 'NZDCAD', 'NZDCHF', 'CADCHF']
    list1 += ['GBPUSD', 'AUDCHF', 'AUDJPY']
    list1 += ['EURUSD', 'USDCHF', 'EURCAD', 'EURGBP', 'GBPCAD', 'GBPJPY', 'AUDCAD', 'AUDNZD', 'CHFJPY']
    list1 += ['AUDUSD', 'NZDUSD', 'EURAUD', 'EURCHF', 'EURNZD', 'GBPAUD', 'GBPNZD', 'CADJPY']

    list2 = ['USD/JPY', 'USD/CAD', 'EUR/JPY', 'GBP/CHF', 'NZD/CAD', 'NZD/CHF', 'CAD/CHF']
    list2 += ['GBP/USD', 'AUD/CHF', 'AUD/JPY']
    list2 += ['EUR/USD', 'USD/CHF', 'EUR/CAD', 'EUR/GBP', 'GBP/CAD', 'GBP/JPY', 'AUD/CAD', 'AUD/NZD', 'CHF/JPY']
    list2 += ['AUD/USD', 'NZD/USD', 'EUR/AUD', 'EUR/CHF', 'EUR/NZD', 'GBP/AUD', 'GBP/NZD', 'CAD/JPY']
    # list1 = ['AUDCHF', 'CADCHF']
    # list2 = ['AUD/CHF', 'CAD/CHF']
    timeframe = 'm30'
    n_prices = 16
    windows = 6
    price = dict()
    mylog_path = 'pytorch_log.log'
    amount = 1
    const_profit = 1
    profit = 3
    loss = 1
    win_ratio = 0.5
    loss_ratio = -0.5
    trailing_step = 1
    delta_close = 90
    model = None
    symbol = ''
    symbol2 = ''
    weight_file = ''
    test_pred = False

    signal.signal(signal.SIGALRM, handler)

    con = connect()
    offer = con.get_offers(kind='dataframe')

    model = resnet18(3)
    model.to(device)
    writeLog(str(datetime.now()))
    init(open_new=False)
    writeLog('Init finish')
    writeLog('-----------')

    while True:
        currenttime = datetime.now()
        if currenttime.second == 0 and currenttime.minute % 30 == 0:
            Update()
            time.sleep(1500)
        time.sleep(1)


# In[ ]:
