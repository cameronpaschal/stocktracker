import pandas as pd
import numpy as np


dia = pd.read_csv("../data/DIA.csv")
qqq = pd.read_csv("../data/QQQ.csv")
spy = pd.read_csv("../data/SPY.csv")

dia["Date"] = pd.to_datetime(dia["Date"], format="%Y-%m-%d")
qqq["Date"] = pd.to_datetime(qqq["Date"], format="%Y-%m-%d")
spy["Date"] = pd.to_datetime(spy["Date"], format="%m/%d/%y")





def CleanData(df, stockName):
    df = df.set_index("Date")
    print(type(df.index))
    df = df.sort_values(by="Date")    
    df = df.resample("ME").last()


    df["ret_1m"] = df["Close"].pct_change()
    df["Ticker"] = stockName
    
    clean_frame = df[["Ticker", "Close", "ret_1m"]]

    return clean_frame

def AddFeatures(df):
    df["ret_1m_lag1"] = df["ret_1m"].shift(1)
    df["ret_3m_lag1"] = (
        df["ret_1m"]
        .rolling(window=3, min_periods=3)
        .sum()
        .shift(1)
    )
    df["ret_12m_lag1"] = (
        df["ret_1m"]
        .rolling(window=12, min_periods=12)
        .sum()
        .shift(1)
    )
    df["vol_3m_lag1"] = df["ret_1m"].rolling(3).std().shift(1)
    df["month_of_year"] = df.index.month
    
    return df

    
dia_clean = CleanData(dia, "DIA")
qqq_clean = CleanData(qqq, "QQQ")
spy_clean = CleanData(spy, "SPY")

dia_clean["ret_1m_lag1"] = dia_clean["ret_1m"].shift(1)
dia_clean["ret_3m_lag1"] = (
    dia_clean["ret_1m"]
    .rolling(window=3, min_periods=3)
    .sum()
    .shift(1)
    )
dia_clean["ret_12m_lag1"] = (
        dia_clean["ret_1m"]
        .rolling(window=12, min_periods=12)
        .sum()
        .shift(1)
    )
dia_clean["vol_3m_lag1"] = dia_clean["ret_1m"].rolling(3).std().shift(1)
dia_clean["month_of_year"] = dia_clean.index.month




# comb = pd.concat([dia_clean, qqq_clean, spy_clean]).sort_index()

# comb["ret_1m_lag1"] = comb["ret_1m"].shift(1)
# comb["ret_3m_lag1"] = comb["ret_1m"].apply(lambda x: x.tail(3).sum())



    
print(dia_clean)




