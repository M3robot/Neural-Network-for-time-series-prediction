# Importing Dependencies
import datetime as dt
import math
import os
import random
import time
import warnings
from datetime import datetime
from tkinter import *

import keras
import matplotlib
import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import scipy
import sklearn as sk
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_finance import candlestick_ohlc
from pandas.plotting import register_matplotlib_converters
from plotly import tools
from scipy import stats
from scipy.optimize import OptimizeWarning
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

register_matplotlib_converters()
matplotlib.use("TkAgg")

# Set font for all graphs
matplotlib.rcParams['font.serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "serif"


# Functions and Classes
class Holder:
    None


# Feature Functions
def detrend_data(prices, method="difference"):
    if method == "difference":
        detrended = prices["Close"][1:] - prices["Close"][:-1].values

    elif method == 'linear':

        x = np.arange(0, len(prices))
        y = prices["Close"].values

        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

        trend = model.predict(x.reshape(-1, 1))
        trend = trend.reshape((len(prices),))

        detrended = prices.Close - trend

    else:
        print("You did not enter a valid method")

    return detrended


def fourier_series(x, a0, a1, b1, w):
    f = a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x)
    return f


def sine_series(x, a0, b1, w):
    f = a0 + b1 * np.sin(w * x)
    return f


def fourier(prices, periods, method="difference"):
    holder_object = Holder()
    results_dict = {}
    plot = False
    detrended = detrend_data(prices, method)

    for i in range(0, len(periods)):
        coeffs = []

        for j in range(periods[i], len(prices) - periods[i]):
            x = np.arange(0, periods[i])
            y = detrended.iloc[j - periods[i]:j]

            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)

                try:
                    res = scipy.optimize.curve_fit(fourier_series, x, y)

                except(RuntimeError, OptimizeWarning):

                    res = np.empty((1, 4))
                    res[0, :] = np.NAN
            if plot is True:
                xt = np.linspace(0, periods[i], 100)
                yt = fourier_series(xt, res[0][0], res[0][1], res[0][2], res[0][3])

                plt.plot(x, y)
                plt.plot(xt, yt, 'r')

                plt.show()

            coeffs = np.append(coeffs, res[0], axis=0)

        # warnings.filterwarnings('ingore',category=np.VisibleDeprecationWarning)

        coeffs = np.array(coeffs).reshape((int(len(coeffs) / 4), 4))

        df = pd.DataFrame(coeffs, index=prices.iloc[periods[i]:-periods[i]].index)
        df.columns = ["a0", "a1", "b1", "w1"]
        df = df.fillna(method="ffill")

        results_dict[periods[i]] = df
    holder_object.coeffs = results_dict

    return holder_object


def sine(prices, periods, method="difference"):
    holder_object = Holder()
    results_dict = {}
    plot = False

    detrended = detrend_data(prices, method)

    for i in range(0, len(periods)):
        coeffs = []

        for j in range(periods[i], len(prices) - periods[i]):

            x = np.arange(0, periods[i])
            y = detrended.iloc[j - periods[i]:j]

            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)

                try:
                    res = scipy.optimize.curve_fit(sine_series, x, y)

                except(RuntimeError, OptimizeWarning):

                    res = np.empty((1, 3))
                    res[0, :] = np.NAN
            if plot is True:
                xt = np.linspace(0, periods[i], 100)
                yt = sine_series(xt, res[0][0], res[0][1], res[0][2])

                plt.plot(x, y)
                plt.plot(xt, yt, 'r')

                plt.show()

            coeffs = np.append(coeffs, res[0], axis=0)

        # warnings.filterwarnings('ingore',category=np.VisibleDeprecationWarning)

        coeffs = np.array(coeffs).reshape((int(len(coeffs) / 3), 3))
        df = pd.DataFrame(coeffs, index=prices.iloc[periods[i]:-periods[i]].index)
        df.columns = ["a0", "b1", "w1"]
        df = df.fillna(method="ffill")

        results_dict[periods[i]] = df

    holder_object.coeffs = results_dict

    return holder_object


def heikenashi(prices, periods):
    holder_object = Holder()
    results_dict = {}

    ha_close = (prices["open"] + prices["high"] + prices["low"] + prices["close"]) / 4

    ha_open = ha_close.copy()
    ha_open.iloc[0] = ha_close.iloc[0]

    ha_high = ha_close.copy()
    ha_low = ha_close.copy()

    for i in range(1, len(prices)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] - ha_close.iloc[i - 1]) / 2
        ha_high.iloc[i] = np.array([prices.high.iloc[i], ha_open.iloc[i], ha_close.iloc[i]]).max()
        ha_low.iloc[i] = np.array([prices.low.iloc[i], ha_open.iloc[i], ha_close.iloc[i]]).min()

    df = pd.concat((ha_open, ha_high, ha_low, ha_close), axis=1)
    df.columns = ["Open", "High", "Low", "Close"]

    df.index = df.index.droplevel(0)

    results_dict[periods[0]] = df
    holder_object.candles = results_dict
    return holder_object


def momentum(prices, periods):
    holder_object = Holder()
    open_dict = {}
    close_dict = {}

    for i in range(0, len(periods)):
        open_dict[periods[i]] = pd.DataFrame(prices.Open.iloc[periods[i]:] - prices.Open.iloc[:-periods[i]].values,
                                             index=prices.iloc[periods[i]:].index)
        close_dict[periods[i]] = pd.DataFrame(
            prices.Close.iloc[periods[i]:] - prices.Close.iloc[:-periods[i]].values,
            index=prices.iloc[periods[i]:].index)
        open_dict[periods[i]].columns = ["Open"]
        close_dict[periods[i]].columns = ["Close"]

    holder_object.open = open_dict
    holder_object.close = close_dict

    return holder_object


def stochastic(prices, periods):
    holder_object = Holder()
    close = {}

    for i in range(0, len(periods)):

        ks = []

        for j in range(periods[i], len(prices) - periods[i]):
            c = prices.Close.iloc[j + 1]
            h = prices.High.iloc[j - periods[i]:j].max()
            low = prices.Low.iloc[j - periods[i]:j].min()

            if h == low:
                k = 0

            else:
                k = 100 * (c - low) / (h - low)

            ks = np.append(ks, k)
        df = pd.DataFrame(ks, index=prices.iloc[periods[i] + 1:-periods[i] + 1].index)
        df.columns = ['k']
        df['D'] = df.k.rolling(3).mean()
        df.dropna()

        close[periods[i]] = df

    holder_object.close = close
    return holder_object


def williams_osc(prices, periods):
    holder_object = Holder()
    close = {}

    for i in range(0, len(periods)):

        rs = []

        for j in range(periods[i], len(prices) - periods[i]):
            c = prices.Close.iloc[j + 1]
            h = prices.High.iloc[j - periods[i]:j].max()
            low = prices.Low.iloc[j - periods[i]:j].min()

            if h == low:
                r = 0

            else:
                r = -100 * (h - c) / (h - low)

            rs = np.append(rs, r)

        df = pd.DataFrame(rs, index=prices.iloc[periods[i] + 1:-periods[i] + 1].index)
        df.columns = ["r"]
        df.dropna()

        close[periods[i]] = df

    holder_object.close = close

    return holder_object


def resample_data(dataframe, timeframe):
    grouped = dataframe.groupby('Symbol')

    if np.any(dataframe.columns == "Volume"):
        ask = grouped["Volume"].resample(timeframe).ohlc()
        resampled = pd.DataFrame(ask)

    elif np.any(dataframe.columns == "close"):
        close = grouped["Close"].resample(timeframe).ohlc()
        high = grouped["High"].resample(timeframe).ohlc()
        low = grouped["Low"].resample(timeframe).ohlc()
        volume = grouped["Volume"].resample(timeframe).ohlc()

        resampled = pd.DataFrame(open_dict)
        resampled["High"] = high
        resampled["Low"] = low
        resampled["Close"] = close
        resampled["Volume"] = volume

    resampled = resampled.dropna()

    return resampled


def proc(prices, periods):
    holder_object = Holder()
    proc_var = {}

    for i in range(0, len(periods)):
        proc_var[periods[i]] = pd.DataFrame((prices.Close.iloc[periods[i]:]) - prices.Close.iloc[:-periods[i]].values
                                            / (prices.Close.iloc[:-periods[i]]).values)
        proc_var[periods[i]].columns = ["Close"]

    holder_object.proc_var = proc_var

    return holder_object


def macd(prices, periods):
    holder_object = Holder()

    ema1 = prices.Close.ewm(span=periods[0]).mean()
    ema2 = prices.Close.ewm(span=periods[1]).mean()

    macd_var = pd.DataFrame(ema1 - ema2)
    macd_var.columns = ["Low"]

    sigmacd = macd_var.rolling(3).mean()
    sigmacd.columns = ["SL"]

    holder_object.line = macd_var
    holder_object.signal = sigmacd

    return holder_object


def cci(prices, periods):
    holder_object = Holder()
    cci_var = {}

    for i in range(0, len(periods)):
        ma = prices.Close.rolling(periods[i]).mean()
        std = prices.Close.rolling(periods[i]).std()

        d = (prices.Close - ma) / std

        cci_var[periods[i]] = pd.DataFrame((prices.Close - ma) / (0.015 * d))
        cci_var[periods[i]].columns = ["Close"]

    holder_object.cci_var = cci_var

    return holder_object


def bollinger(prices, periods, deviations):
    holder_object = Holder()
    boll = {}

    for i in range(0, len(periods)):
        mid = prices.Close.rolling(periods[i]).mean()
        std = prices.Close.rolling(periods[i]).std()

        upper = mid + deviations * std
        lower = mid - deviations * std

        df = pd.concat((upper, mid, lower), axis=1)
        df.columns = ["Upper", "Mid", "Lower"]

        boll[periods[i]] = df

    holder_object.bands = boll

    return holder_object


def paverage(prices, periods):
    holder_object = Holder()
    avs = {}

    for i in range(0, len(periods)):
        avs[periods[i]] = pd.DataFrame(prices[["Open", "High", "Low", "Close"]].rolling(periods[i]).mean())

    holder_object.avs = avs

    return holder_object


def slopes(prices, periods):
    holder_object = Holder()
    slope = {}

    for i in range(0, len(periods)):

        ms = []

        for j in range(periods[i], len(prices) - periods[i]):
            y = prices.High.iloc[j - periods[i]:j].values
            x = np.arange(0, len(y))

            res = stats.linregress(x, y=y)
            m = res.slope

            ms = np.append(ms, m)

        ms = pd.DataFrame(ms, index=(prices.iloc[periods[i]:-periods[i]]).index)
        ms.columns = ["High"]

        slope[periods[i]] = ms

    holder_object.slope = slope
    return holder_object


def wadl(prices, periods):
    holder_object = Holder()
    results_dict = {}

    for i in range(0, len(periods)):
        # Williams Accumulation Distribution Function
        wad = []

        for j in range(periods[i], len(prices) - periods[i]):

            trh = np.array([prices.High.iloc[j]], prices.Close.iloc[j - 1]).max()
            trl = np.array([prices.Low.iloc[j]], prices.Close.iloc[j - 1]).min()

            if prices.Close.iloc[j] > prices.Close.iloc[j - 1]:
                pm = prices.Close.iloc[j] - trl

            elif prices.Close.iloc[j] < prices.Close.iloc[j - 1]:
                pm = prices.Close.iloc[j] - trh

            elif prices.Close.iloc[j] == prices.Close.iloc[j - 1]:
                pm = 0

            ad = pm * prices.Volume.iloc[j]
            wad = np.append(wad, ad)

        wad = wad.cumsum()
        wad = pd.DataFrame(wad, index=prices.iloc[periods[i]:-periods[i]].index)
        wad.columns = ["Close"]

        results_dict[periods[i]] = wad
        holder_object.wadl = results_dict

    return holder_object


# Collect Features / Create MasterFrame.csv
def collect(csv):
    path = r"Stock Data\.csv"
    path = path[:11] + csv + path[11:]
    print(path)
    data = pd.read_csv(path)

    drop_list = ["Date", "Adj Close"]
    data.columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    data = data.set_index(pd.to_datetime(data.Date))
    data = data.drop(drop_list, 1)
    data.columns = ["Open", "High", "Low", "Close", "Volume"]

    prices = data.drop_duplicates(keep=False)

    # Create period list
    momentum_key = [3, 4, 5, 8, 9, 10]
    stochastic_key = [3, 4, 5, 8, 9, 10]
    williams_key = [6, 7, 8, 9, 10]
    proc_key = [12, 13, 14, 15]
    wadl_key = [15]
    adosc_key = [2, 3, 4, 5]
    macd_key = [15, 30]
    cci_key = [15]
    bollinger_key = [15]
    heiken_ashi_key = [15]
    paverage_key = [2]
    slope_key = [3, 4, 5, 10, 20, 30]
    fourier_key = [10, 20, 30]
    sine_key = [5, 6]

    key_list = [momentum_key, stochastic_key, williams_key, proc_key, wadl_key, macd_key, cci_key, bollinger_key,
                heiken_ashi_key, paverage_key, slope_key, fourier_key, sine_key]

    # Creating Dictionaries
    momentum_dict = momentum(prices, momentum_key)
    print("Momentum Calculated  1")
    stochastic_dict = stochastic(prices, stochastic_key)
    print("Stochastic Calculated  2")
    williams_dict = williams_osc(prices, williams_key)
    print("Williams Oscillator Calculated  3")
    proc_dict = proc(prices, proc_key)
    print("Price Rate of Change Calculated  4")
    wadl_dict = wadl(prices, wadl_key)
    print("Williams Accumulation Distribution Calculated  5")
    macd_dict = macd(prices, macd_key)
    print("Moving Average Convergence Divergence Calculated  6")
    cci_dict = cci(prices, cci_key)
    print("Commodity Channel Index  7")
    bollinger_dict = bollinger(prices, bollinger_key, 2)
    print("Bollinger Calculated  8")

    hka_prices = prices.copy()
    hka_prices["Symbol"] = "MSFT"

    hka = resample_data(hka_prices, "15H")

    heiken_dict = heikenashi(hka, heiken_ashi_key)
    print("Heiken Ashi Candles Calculated  9")
    paverage_dict = paverage(prices, paverage_key)
    print("Price Average Calculated  10")
    slope_dict = slopes(prices, slope_key)
    print("Slopes Calculated  11")
    fourier_dict = fourier(prices, fourier_key)
    print("Fourier Series Calculated  12")
    sine_dict = sine(prices, sine_key)
    print("Sine Series  13")

    dict_list = [momentum_dict.close, stochastic_dict.close, williams_dict.close, proc_dict.proc_var, wadl_dict.wadl,
                 macd_dict.line, cci_dict.cci_var, bollinger_dict.bands, heiken_dict.candles, paverage_dict.avs,
                 slope_dict.slope, fourier_dict.coeffs, sine_dict.coeffs]

    # List of Base Column Names

    col_feat = ["Momentum", "Stochastic", "Williams %R", "PROC", "WADL", "MACD", "CCI", "Bollinger", "Heiken",
                "PAverage", "Slope", "Fourier", "Sine"]

    # Create and Populate the Master Frame

    master_frame = pd.DataFrame(index=prices.index)

    for i in range(0, len(dict_list)):

        if col_feat[i] == "MACD":
            col_id = col_feat[i] + str(key_list[5][1]) + str(key_list[5][1])
            master_frame[col_id] = dict_list[i]

        else:
            for j in key_list[i]:
                for k in list(dict_list[i][j]):
                    col_id = col_feat[i] + str(j) + str(k)

                    master_frame[col_id] = dict_list[i][j][k]

    threshold = round(0.9 * len(master_frame))

    master_frame[["Open", "High", "Low", "Close", "Volume"]] = prices[["Open", "High", "Low", "Close", "Volume"]]

    # Heiken Ashi is resampled => empty data in between

    master_frame.Heiken15Open = master_frame.Heiken15Open.fillna(method="ffill")
    master_frame.Heiken15Close = master_frame.Heiken15Close.fillna(method="ffill")
    master_frame.Heiken15Low = master_frame.Heiken15Low.fillna(method="ffill")
    master_frame.Heiken15High = master_frame.Heiken15High.fillna(method="ffill")

    master_frame_cleaned = master_frame.copy()

    master_frame_cleaned = master_frame_cleaned.dropna(axis=1, thresh=threshold)
    master_frame_cleaned = master_frame_cleaned.dropna(axis=0)

    master_frame_cleaned.to_csv("MasterFrame.csv")

    print("Completed Feature Calculations")


# Prepare and Graph Data. Train Model and Predict based on model
def prepare_data():
    holder_object = Holder()

    hyperparams_df = pd.read_csv("Settings.csv")

    batch_percentage = float(hyperparams_df.at[0, "Batch"])
    test_size = float(hyperparams_df.at[0, "Test Size"])
    epochs = int(hyperparams_df.at[0, 'Epochs'])
    decay = float(hyperparams_df.at[0, "Decay"])
    learning_rate = float(hyperparams_df.at[0, "Learning Rate"])
    patience = int(hyperparams_df.at[0, "Patience"])
    predict_days = int(hyperparams_df.at[0, "Predict Days"])
    lstm_neurons = int(hyperparams_df.at[0, "LSTM Neurons"])
    dense1_neurons = int(hyperparams_df.at[0, "Dense 1 Neurons"])
    dense2_neurons = int(hyperparams_df.at[0, "Dense 2 Neurons"])
    dense3_neurons = int(hyperparams_df.at[0, "Dense 3 Neurons"])
    activation_function = str(hyperparams_df.at[0, "Activation Function"])

    prediction_col = "Forecast"
    predict_values = "Close"

    df = pd.read_csv("MasterFrame.csv")
    df.fillna(0, inplace=True)
    df = df.set_index(pd.to_datetime(df.Date))
    df = df.drop(["Date"], 1)

    data = df.copy()
    data[prediction_col] = data[predict_values].shift(-predict_days)
    data.dropna(inplace=True)

    datax = data.drop([prediction_col], 1)
    sc_data = normalize_data(datax)

    x = np.array(sc_data.values)
    y = np.array(data[prediction_col].values).reshape(-1, 1)

    scaler = StandardScaler()
    y = scaler.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)

    x_train_lmse = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test_lmse = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    batch_size = int(math.ceil(batch_percentage * len(data[prediction_col])))
    opt = Adam(lr=learning_rate, decay=decay)

    # Build Model    
    model = Sequential()
    model.add(LSTM(lstm_neurons, input_shape=(x_train_lmse.shape[1], x_train_lmse.shape[2]),
                   activation=activation_function, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(dense1_neurons, activation=activation_function))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(Dense(dense2_neurons, activation=activation_function))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(dense3_neurons, activation=activation_function))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=opt, metrics=["mse"])

    callback = [EarlyStopping(patience=patience,
                              verbose=1),
                ModelCheckpoint(filepath="best_model.h5",
                                verbose=1,
                                save_best_only=True)]

    history = model.fit(x_train_lmse, y_train,
                        callbacks=callback,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=(x_test_lmse, y_test))

    error_df = pd.DataFrame({"error": history.history["loss"],
                             "val_error": history.history["val_loss"]})

    model.load_weights("best_model.h5")
    model.compile(loss="mse", optimizer=opt)

    new_values = normalize_data(df)
    new_values = np.array(new_values.values)
    new_values = new_values.reshape(new_values.shape[0], new_values.shape[1], 1)

    ynew = model.predict(new_values)
    ynew = scaler.inverse_transform(ynew)
    ynewdf = pd.DataFrame(ynew, index=df.index)

    ynewdf.columns = [prediction_col]
    ynewdf["Rolling Average"] = ynewdf[prediction_col].rolling(predict_days).mean()
    ynewdf["Real Values"] = df[predict_values].tolist()

    holder_object.original = go.Scatter(x=df.index, y=ynewdf["Real Values"], name="Real Values")
    holder_object.predicted = go.Scatter(x=ynewdf.index, y=ynewdf[prediction_col], name="Predicted Values")
    holder_object.moving_average = go.Scatter(x=ynewdf.index, y=ynewdf["Rolling Average"],
                                              name="Predicted Values Moving Average")

    ynewdf.to_csv("Prediction.csv")
    error_df.to_csv("Error.csv")

    return holder_object


def graph_prediction(original, predicted, ma):
    fig = tools.make_subplots(rows=1, cols=1, shared_xaxes=False)
    fig.append_trace(original, 1, 1)
    fig.append_trace(ma, 1, 1)
    fig.append_trace(predicted, 1, 1)

    py.offline.plot(fig, "Real vs predicted Values.html")


def normalize_data(df):
    result = df.copy()
    for feature_name in df.columns:
        mean_value = df[feature_name].mean()
        standard_deviation_value = df[feature_name].std()
        result[feature_name] = (df[feature_name] - mean_value) / standard_deviation_value
    return result


# create_settings_csv(0.0125, 0.2, 750, 1e-5, 1.75e-4, 25, 30, 100, 32, 32, 16, "relu", "MSFT")
def create_settings_csv(csv, batch_percentage=0.125, test_size=0.2, epochs=750, decay=1e-5, learning_rate=1e-4,
                        patience=25, predict_days=30, lstm_neurons=100, dense1_neurons=32, dense2_neurons=32,
                        dense3_neurons=16, activation="relu"):
    settings_df = pd.DataFrame({"Batch": batch_percentage,
                                "Test Size": test_size,
                                "Epochs": epochs,
                                "Decay": decay,
                                "Learning Rate": learning_rate,
                                "Patience": patience,
                                "Predict Days": predict_days,
                                "LSTM Neurons": lstm_neurons,
                                "Dense 1 Neurons": dense1_neurons,
                                "Dense 2 Neurons": dense2_neurons,
                                "Dense 3 Neurons": dense3_neurons,
                                "Activation Function": activation,
                                "CSV": csv}, index=[0])
    settings_df.to_csv("Settings.csv")


class LoginWindow:
    def __init__(self):
        self.uname = "JohnDoe"
        self.pword = "Stocks"

        self.rootA = Tk()
        self.rootA.title("Login")

        self.userL = Label(self.rootA, text='Username')
        self.passL = Label(self.rootA, text="Password")

        self.passL.grid(row=1, sticky=E, pady=5, padx=10)
        self.userL.grid(row=0, sticky=E, pady=5, padx=10)

        self.userE = Entry(self.rootA)
        self.passE = Entry(self.rootA, show='*')

        self.passE.grid(column=1, row=1, pady=5, padx=10)
        self.userE.grid(column=1, row=0, pady=5, padx=10)

        self.loginB = Button(self.rootA, text="Login", command=self.check_login)
        self.loginB.grid(sticky=E, column=1, row=2, padx=15, pady=12)

        self.rootA.mainloop()

    def check_login(self):

        if self.userE.get() == self.uname and self.passE.get() == self.pword:
            self.rootA.destroy()
            Dashboard()
        else:
            r = Tk()
            r.title("Oops")
            r.geometry('150x50')
            rlbl = Label(r, text='\n[!] Invalid Login')
            rlbl.pack()
            r.mainloop()


class Dashboard:
    def __init__(self):
        self.roots = Tk()
        self.roots.title('Dashboard')
        self.roots.state('zoomed')

        # Making the Menus
        main_menu = Menu(self.roots)
        self.roots.config(menu=main_menu)
        self.clock = Label(self.roots)
        self.clock.grid(column=0, row=0, columnspan=8)

        main_sub_menu = Menu(main_menu)
        main_menu.add_cascade(label="Main", menu=main_sub_menu)
        main_sub_menu.add_command(label="Dashboard")
        main_sub_menu.add_command(label="Settings", command=self.go_to_settings)
        main_sub_menu.add_command(label="More Graphs", command=self.go_to_more_graphs)
        main_sub_menu.add_separator()
        main_sub_menu.add_command(label="Exit", command=self.end_program)

        # Getting Data
        error_df = pd.read_csv("Error.csv")
        most_current_date = pd.read_csv("MasterFrame.csv")
        masterframe_df = most_current_date.copy()
        masterframe_df = masterframe_df.set_index(pd.to_datetime(masterframe_df.Date))
        most_current_date = most_current_date.Date[most_current_date.index[-1]]
        most_current_date = dt.datetime.strptime(most_current_date, "%Y-%m-%d")
        year_mark = most_current_date - dt.timedelta(weeks=52)

        history_df = masterframe_df.loc[year_mark:]
        history_vals = history_df["Close"].tolist()

        prediction_df = pd.read_csv("Prediction.csv")
        prediction_df = prediction_df.set_index(pd.to_datetime(prediction_df.Date))
        prediction_df = prediction_df.loc[year_mark:]
        forecast = prediction_df["Forecast"].tolist()
        rolling_average = prediction_df["Rolling Average"].tolist()

        # Making the Graphs
        fig = matplotlib.figure.Figure(figsize=(15, 3.25))
        ax = fig.add_subplot(121)
        ax.set_title("Validation Error vs Epochs")
        ax.plot(error_df.index, error_df["error"].tolist(), "r", label="Error")
        ax.plot(error_df.index, error_df["val_error"].tolist(), "b", label="Validation Error")
        ax.legend(loc="upper right")

        self.canvas = FigureCanvasTkAgg(fig, master=self.roots)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=8, pady=10, padx=10)

        ax2 = fig.add_subplot(122)
        ax2.set_title("Stock History\n(Last 52 Weeks)")
        ax2.plot(history_df.index, history_vals, "b")

        self.fig2 = matplotlib.figure.Figure(figsize=(15, 3.25))
        ax3 = self.fig2.add_subplot(111)
        ax3.set_title("Predictions vs Real")
        ax3.plot(prediction_df.index, history_vals, "b", label="Real Values")
        ax3.plot(prediction_df.index, forecast, "g", label="Prediction")
        ax3.plot(prediction_df.index, rolling_average, "r", label="Prediction Rolling Average")
        ax3.legend(loc="upper right")

        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.roots)
        self.canvas2.get_tk_widget().grid(row=4, column=0, columnspan=8, pady=10, padx=10)
        self.tick()
        self.roots.mainloop()

    def tick(self):
        time_string = time.strftime("%I:%M:%S")
        self.clock.config(text=time_string)
        self.clock.after(200, self.tick)

    def go_to_settings(self):
        self.roots.destroy()
        Settings()

    def go_to_more_graphs(self):
        self.roots.destroy()
        MoreGraphs()

    def end_program(self):
        self.roots.destroy()


class MoreGraphs:
    def __init__(self):
        self.roots = Tk()
        self.roots.title('Dashboard')
        self.roots.state('zoomed')

        # Making the Menus
        main_menu = Menu(self.roots)
        self.roots.config(menu=main_menu)
        self.clock = Label(self.roots)
        self.clock.grid(column=0, row=0, columnspan=8)

        main_sub_menu = Menu(main_menu)
        main_menu.add_cascade(label="Main", menu=main_sub_menu)
        main_sub_menu.add_command(label="Dashboard", command=self.go_to_dash)
        main_sub_menu.add_command(label="Settings", command=self.go_to_settings)
        main_sub_menu.add_command(label="More Graphs")
        main_sub_menu.add_separator()
        main_sub_menu.add_command(label="Exit", command=self.end_program)

        # Get Data
        masterframe_df = pd.read_csv("MasterFrame.csv")

        most_current_date = masterframe_df.copy()
        most_current_date = most_current_date["Date"][most_current_date.index[-1]]
        most_current_date = dt.datetime.strptime(most_current_date, "%Y-%m-%d")
        year_mark = most_current_date - dt.timedelta(weeks=52)

        masterframe_df["Date"] = pd.to_datetime(masterframe_df["Date"])
        masterframe_df = masterframe_df.set_index(masterframe_df["Date"])
        masterframe_df = masterframe_df.drop("Date", 1)

        history_df = masterframe_df.loc[year_mark:]

        moving_average_graph = history_df.Close.rolling(center=False, window=30).mean()

        williams_accumulation = wadl(history_df, [15])
        williams_accumulation_graph = williams_accumulation.wadl[15]

        momentum_values = momentum(history_df, [10])
        momentum_graph = momentum_values.close[10]

        stochastic_results = stochastic(history_df, [14])
        stochastic_graph = stochastic_results.close[14]

        williams_osc_results = williams_osc(history_df, [15])
        williams_osc_graph = williams_osc_results.close[15]

        price_rate_results = proc(history_df, [13])
        price_rate_graph = price_rate_results.proc_var[13]

        # Create graphs
        fig = matplotlib.figure.Figure(figsize=(12, 7))
        ax = fig.add_subplot(711)
        ax.set_title("History")
        ax.plot(history_df.index, history_df["Close"].tolist(), "r", label="Historical Values")
        ax.plot(history_df.index, moving_average_graph, "b", label="Moving Average")
        ax.legend(loc="upper right")

        ax2 = fig.add_subplot(713)
        ax2.set_title("Williams Accumulation Distribution Line")
        ax2.plot(history_df.index[:len(williams_accumulation_graph)], williams_accumulation_graph, "r",
                 label="Williams Accumulation Distribution")
        ax2.axhline(y=0, color="k", dashes=[4, 2])
        ax2.legend(loc="upper right")

        ax3 = fig.add_subplot(715)
        ax3.set_title("Momentum")
        ax3.plot(history_df.index[:len(momentum_graph)], momentum_graph, "r", label="Momentum")
        ax3.axhline(y=0, color="k", dashes=[4, 2])
        ax3.legend(loc="upper right")

        ax4 = fig.add_subplot(717)
        ax4.set_title("Miscellaneous")
        ax4.plot(history_df.index[:len(stochastic_graph)], stochastic_graph, "r", label="Stochastic Line")
        ax4.plot(history_df.index[:len(price_rate_graph)], price_rate_graph, "b", label="Price Rate of Change")
        ax4.plot(history_df.index[:len(williams_osc_graph)], williams_osc_graph, "g", label="Williams Oscillator")
        ax4.legend(loc="upper right")

        self.canvas = FigureCanvasTkAgg(fig, master=self.roots)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=2)

        self.roots.mainloop()

    def go_to_settings(self):
        self.roots.destroy()
        Settings()

    def go_to_dash(self):
        self.roots.destroy()
        Dashboard()

    def end_program(self):
        self.roots.destroy()


class Settings:
    def __init__(self):
        self.roots = Tk()
        self.roots.title('Dashboard')
        self.roots.state('zoomed')

        # Making the Menus
        main_menu = Menu(self.roots)
        self.roots.config(menu=main_menu)
        self.clock = Label(self.roots)
        self.clock.grid(column=0, row=0, columnspan=8)

        main_sub_menu = Menu(main_menu)
        main_menu.add_cascade(label="Main", menu=main_sub_menu)
        main_sub_menu.add_command(label="Dashboard", command=self.go_to_dash)
        main_sub_menu.add_command(label="Settings")
        main_sub_menu.add_command(label="More Graphs", command=self.go_to_more_graphs)
        main_sub_menu.add_separator()
        main_sub_menu.add_command(label="Exit", command=self.end_program)

        # Getting Data
        self.settings_df = pd.read_csv("Settings.csv")
        self.preset_batch_percentage = self.settings_df.at[0, "Batch"]
        self.preset_test_size = self.settings_df.at[0, "Test Size"]
        self.preset_epochs = self.settings_df.at[0, 'Epochs']
        self.preset_decay = self.settings_df.at[0, "Decay"]
        self.preset_learning_rate = self.settings_df.at[0, "Learning Rate"]
        self.preset_patience = self.settings_df.at[0, "Patience"]
        self.preset_predict_days = self.settings_df.at[0, "Predict Days"]
        self.preset_lstm_neurons = self.settings_df.at[0, "LSTM Neurons"]
        self.preset_dense1_neurons = self.settings_df.at[0, "Dense 1 Neurons"]
        self.preset_dense2_neurons = self.settings_df.at[0, "Dense 2 Neurons"]
        self.preset_dense3_neurons = self.settings_df.at[0, "Dense 3 Neurons"]

        lstm_current_neurons = "Current number: {} Neurons".format(self.preset_lstm_neurons)
        self.lstm_main_label = Label(self.roots, text="LSTM Layer", font=("Times New Roman", 12, "bold"))
        self.lstm_current_label = Label(self.roots, text=lstm_current_neurons, font=("Times New Roman", 12))
        self.lstm_entry_label = Label(self.roots, text="New number of neurons in LSTM layer:",
                                      font=("Times New Roman", 12))
        self.lstm_entry = Entry(self.roots)

        self.lstm_main_label.grid(row=2, column=0, pady=5, padx=10, columnspan=4, sticky=W)
        self.lstm_current_label.grid(row=3, column=0, pady=5, padx=10, columnspan=4)
        self.lstm_entry_label.grid(row=4, column=0, pady=5, padx=5, sticky=W, columnspan=2)
        self.lstm_entry.grid(row=4, column=3, pady=5, padx=5, sticky=E, columnspan=2)

        dense1_current_neurons = "Current number: {} Neurons".format(self.preset_dense1_neurons)
        self.dense1_main_label = Label(self.roots, text="Dense Layer", font=("Times New Roman", 12, "bold"))
        self.dense1_current_label = Label(self.roots, text=dense1_current_neurons, font=("Times New Roman", 12))
        self.dense1_entry_label = Label(self.roots, text="New number of neurons in Dense layer:",
                                        font=("Times New Roman", 12))
        self.dense1_entry = Entry(self.roots)

        self.dense1_main_label.grid(row=5, column=0, pady=5, padx=10, columnspan=4, sticky=W)
        self.dense1_current_label.grid(row=6, column=0, pady=5, padx=10, columnspan=4)
        self.dense1_entry_label.grid(row=7, column=0, pady=5, padx=5, sticky=W, columnspan=2)
        self.dense1_entry.grid(row=7, column=3, pady=5, padx=5, sticky=E, columnspan=2)

        dense2_current_neurons = "Current number: {} Neurons".format(self.preset_dense2_neurons)
        self.dense2_main_label = Label(self.roots, text="Dense Layer", font=("Times New Roman", 12, "bold"))
        self.dense2_current_label = Label(self.roots, text=dense2_current_neurons, font=("Times New Roman", 12))
        self.dense2_entry_label = Label(self.roots, text="New number of neurons in Dense layer:",
                                        font=("Times New Roman", 12))
        self.dense2_entry = Entry(self.roots)

        self.dense2_main_label.grid(row=8, column=0, pady=5, padx=10, columnspan=4, sticky=W)
        self.dense2_current_label.grid(row=9, column=0, pady=5, padx=10, columnspan=4)
        self.dense2_entry_label.grid(row=10, column=0, pady=5, padx=5, sticky=W, columnspan=2)
        self.dense2_entry.grid(row=10, column=3, pady=5, padx=5, sticky=E, columnspan=2)

        dense3_current_neurons = "Current number: {} Neurons".format(self.preset_dense3_neurons)
        self.dense3_main_label = Label(self.roots, text="Dense Layer", font=("Times New Roman", 12, "bold"))
        self.dense3_current_label = Label(self.roots, text=dense3_current_neurons, font=("Times New Roman", 12))
        self.dense3_entry_label = Label(self.roots, text="New number of neurons in Dense layer:",
                                        font=("Times New Roman", 12))
        self.dense3_entry = Entry(self.roots)

        self.dense3_main_label.grid(row=11, column=0, pady=5, padx=10, columnspan=4, sticky=W)
        self.dense3_current_label.grid(row=12, column=0, pady=5, padx=10, columnspan=4)
        self.dense3_entry_label.grid(row=13, column=0, pady=5, padx=5, sticky=W, columnspan=2)
        self.dense3_entry.grid(row=13, column=3, pady=5, padx=5, sticky=E, columnspan=2)

        self.apply_button = Button(self.roots, text="Apply", command=self.get_new_vals)
        self.apply_button.grid(row=14, column=5, padx=10, pady=5)
        self.cancel_button = Button(self.roots, text="Cancel", command=self.go_to_dash)

        # Run mainloop
        self.roots.mainloop()

    def get_new_vals(self):
        self.new_lstm_val = self.lstm_entry.get()
        self.new_dense1_val = self.dense1_entry.get()
        self.new_dense2_val = self.dense2_entry.get()
        self.new_dense3_val = self.dense3_entry.get()

        if self.new_lstm_val == "":
            self.new_lstm_val = self.preset_lstm_neurons
        
        if self.new_dense1_val == "":
            self.new_dense1_val = self.preset_dense1_neurons
            
        if self.new_dense2_val == "":
            self.new_dense2_val = self.preset_dense2_neurons
            
        if self.new_dense3_val == "":
            self.new_dense3_val = self.preset_dense3_neurons

        create_settings_csv("FB", lstm_neurons=self.new_lstm_val, dense1_neurons=self.new_dense1_val,
                            dense2_neurons=self.new_dense2_val, dense3_neurons=self.new_dense3_val)

    def go_to_more_graphs(self):
        self.roots.destroy()
        MoreGraphs()

    def go_to_dash(self):
        self.roots.destroy()
        Dashboard()

    def end_program(self):
        self.roots.destroy()


# results = prepare_data()
# graph_prediction(results.original, results.predicted, results.moving_average)
Settings()
