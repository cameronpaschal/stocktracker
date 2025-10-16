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
    
dia_clean = CleanData(dia, "DIA")
qqq_clean = CleanData(qqq, "QQQ")
spy_clean = CleanData(spy, "SPY")


comb = pd.concat([dia_clean, qqq_clean, spy_clean]).sort_index()



    
print(comb.head)


