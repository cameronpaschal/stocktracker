import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import List
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from scipy.stats import norm
from pathlib import Path


#cleans data and adds the stock name / ticker
def CleanData(df, stockName):
    df = df.set_index("Date")
    print(type(df.index))
    df = df.sort_values(by="Date")    
    df = df.resample("ME").last()


    df["ret_1m"] = df["Close"].pct_change()
    df["Ticker"] = stockName
    
    clean_frame = df[["Ticker", "Close", "ret_1m"]]

    return clean_frame


#adds features to the dataframe to be used in model calculations
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
    df = df.dropna(subset=["ret_1m",  "ret_1m_lag1" , "ret_3m_lag1" , "ret_12m_lag1" , "vol_3m_lag1",  "month_of_year"])
    
    return df

#adds the targets and if they 

def AddTargets(df):
    df["y_ret_next"] = df["ret_1m"].shift(-1)
    df["y_up"] = (df["y_ret_next"] > 0).astype(int)
    return df.dropna(subset=["y_ret_next", "y_up"])

def AddYUp(df):
    df["y_ret_next"] = df["ret_1m"].shift(-1)
    df["y_up"] = (df["y_ret_next"] > 0).astype(int)

    return df.dropna(subset=["y_ret_next", "y_up"])
    
def GenerateFolds(df):
    folds = []
    max_trainable_date = df.index.max()
    min_usable_date = df.index.min()
    window_length = 10
    first_train_year = min_usable_date.year + window_length
    last_year = max_trainable_date.year
    

    for Y in range(first_train_year+1, last_year +1):
        train_start = pd.to_datetime({"year": [Y - window_length], "month": [1],"day": [31]}).iloc[0]
        train_end = pd.to_datetime({"year": [Y -1], "month": [12],"day": [31]}).iloc[0]
        val_start = pd.to_datetime({"year": [Y], "month": [1],"day": [31]}).iloc[0]
        val_end = pd.to_datetime({"year": [Y], "month": [12],"day": [31]}).iloc[0]
        if val_end <= max_trainable_date and train_start >= min_usable_date:
            folds.append((train_start, train_end, val_start, val_end))
            
    return folds
    
def AssessFold(df, fold):
    start_train = fold[0]
    end_train = fold[1]
    start_val = fold[2]
    end_val = fold[3]

    numeric_features = ["ret_1m_lag1" , "ret_3m_lag1" , "ret_12m_lag1" , "vol_3m_lag1"]
    categorical_features = ["month_of_year"]
    label = "y_up"
    expected_month_cols = ["mo_1", "mo_2", "mo_3", "mo_4", "mo_5", "mo_6", "mo_7", "mo_8", "mo_9", "mo_10", "mo_11", "mo_12"]

    df_t = df.loc[start_train:end_train]
    df_v = df.loc[start_val:end_val]

    df_num_t = df_t[numeric_features]
    df_num_v = df_v[numeric_features]

    scaler = StandardScaler()
    scaler.fit(df_num_t)

    df_num_scaled_t = scaler.transform(df_num_t)
    df_num_scaled_v = scaler.transform(df_num_v)

    df_num_scaled_t = pd.DataFrame(df_num_scaled_t, index=df_num_t.index, columns=numeric_features)
    df_num_scaled_v = pd.DataFrame(df_num_scaled_v, index=df_num_v.index, columns=numeric_features)



    #dummy creation
    df_t_dummies = pd.get_dummies(df_t["month_of_year"], prefix="mo", dtype=int)
    df_t_dummies = df_t_dummies.reindex(columns=expected_month_cols, fill_value=0)

    df_v_dummies = pd.get_dummies(df_v["month_of_year"], prefix="mo", dtype=int)
    df_v_dummies = df_v_dummies.reindex(columns=expected_month_cols, fill_value=0)


    #concat the dummies and the numeric after it has been scaled
    x_train = pd.concat([df_num_scaled_t,df_t_dummies], axis=1, join="inner")
    x_val = pd.concat([df_num_scaled_v, df_v_dummies], axis=1, join="inner")

    y_train = df_t[label]
    y_val = df_v[label]


    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(x_train,y_train)

    p_up = clf.predict_proba(x_val)[:, 1]
    return p_up  

def append_fold_predictions(
    pred_rows_list: List[pd.DataFrame],
    dates, tickers, p_up, y_up, y_ret_next,
    fold_id, train_end, val_start, val_end
) -> None:
    dfp = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "ticker": tickers,
        "p_up": np.asarray(p_up, dtype=float),
        "y_up": np.asarray(y_up, dtype=int),
        "y_ret_next": np.asarray(y_ret_next, dtype=float),
    })
    dfp["fold_id"] = fold_id
    dfp["train_end"] = pd.to_datetime(train_end)
    dfp["val_start"] = pd.to_datetime(val_start)
    dfp["val_end"] = pd.to_datetime(val_end)
    dfp["year"] = dfp["date"].dt.year

    if dfp.isna().any().any():
        missing_cols = dfp.columns[dfp.isna().any()].tolist()
        raise ValueError(f"NaNs in appended predictions: {missing_cols}")

    pred_rows_list.append(dfp)  
    
    
def CombineFolds(df, folds, pred_rows_list):
    for (start_train, end_train, start_val, end_val) in folds:
    # 1) run your fold assessment (gets p_up aligned to validation rows)
        p_up = AssessFold(df, (start_train, end_train, start_val, end_val))

        # 2) grab the validation slice for metadata/ground truth
        df_v = df.loc[start_val:end_val]

        # (safety) ensure lengths match
        if len(p_up) != len(df_v):
            raise ValueError(f"Length mismatch: p_up({len(p_up)}) vs df_v({len(df_v)})")

        # 3) prepare the pieces append_fold_predictions expects
        dates_val   = df_v.index
        tickers_val = df_v["Ticker"] if "Ticker" in df_v.columns else "QQQ"
        y_up_val    = df_v["y_up"]
        y_next_val  = df_v["y_ret_next"]

        # 4) append this fold’s predictions
        append_fold_predictions(
            pred_rows_list,
            dates=dates_val,
            tickers=tickers_val,
            p_up=p_up,
            y_up=y_up_val,
            y_ret_next=y_next_val,
            fold_id=f"{start_val.year}", 
            train_end=end_train,
            val_start=start_val,
            val_end=end_val
        )

def combine_predictions(pred_rows_list: List[pd.DataFrame]) -> pd.DataFrame:
    if not pred_rows_list:
        return pd.DataFrame(columns=[
            "date","ticker","p_up","y_up","y_ret_next","fold_id","train_end","val_start","val_end","year"
        ])
    out = pd.concat(pred_rows_list, ignore_index=True)
    return out.sort_values(["date","ticker"], kind="stable")

def interpret_predictions(pred_rows_list):
    pred_df = combine_predictions(pred_rows_list)

    # 4) Quick metrics on this fold (optional now; will be more meaningful after many folds)
    y_pred = (pred_df["p_up"] >= 0.5).astype(int)
    acc = accuracy_score(pred_df["y_up"], y_pred)
    ic  = spearmanr(pred_df["p_up"], pred_df["y_ret_next"]).correlation
    print(f"[ALL FOLDS] Accuracy={acc:.3f}  IC={ic:.3f}")
    
def PredictNextMonth(df, name): 

    last_complete_date = df.index.max()
    final_df = df.loc[:last_complete_date]

    numeric_features = ["ret_1m_lag1","ret_3m_lag1","ret_12m_lag1","vol_3m_lag1"]
    expected_month_cols = [f"mo_{m}" for m in range(1,13)]


    final_scaler = StandardScaler().fit(final_df[numeric_features])
    final_num_scaled = pd.DataFrame(
        final_scaler.transform(final_df[numeric_features]),
        index=final_df.index, columns=numeric_features
    )
    final_dummies = pd.get_dummies(final_df["month_of_year"], prefix="mo", dtype=int) \
                    .reindex(columns=expected_month_cols, fill_value=0)
    x_final = pd.concat([final_num_scaled, final_dummies], axis=1, join="inner")
    y_final = final_df["y_up"]


    final_clf = LogisticRegression(solver="lbfgs", max_iter=1000).fit(x_final, y_final)
    p_up_current = float(final_clf.predict_proba(x_final.iloc[[-1]])[:, 1])
    print(f"Current probability for {name} next month: {p_up_current:.3f}")

def ScenarioPredictNextMonth(y_next_series: pd.Series, recent_window: int = 36,lower=-0.02, upper=0.02):
    hist = y_next_series.dropna().iloc[-recent_window:]
    if len(hist) < 12:
        hist = y_next_series.dropna()
    mu = float(hist.mean())
    sd = float(hist.std(ddof=1))
    if sd == 0 or np.isnan(sd):
        return 0.0, 1.0, 0.0
    p_bear = norm.cdf(lower, loc=mu, scale=sd)
    p_bull = 1.0 - norm.cdf(upper, loc=mu, scale=sd)
    p_base = max(0.0, 1.0 - p_bear - p_bull)
    return p_bear, p_base, p_bull
    

def FullFlow(file, dateFormat):
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"], format=dateFormat)
    name = Path(file).stem[-3:]  
    df_clean = df_clean = CleanData(df, name)
    df_c_f = AddFeatures(df_clean)
    df_c_f = AddYUp(df_c_f)
    folds = GenerateFolds(df_c_f)
    pred_rows_list = []
    CombineFolds(df_c_f, folds, pred_rows_list)
    interpret_predictions(pred_rows_list)
    print()
    PredictNextMonth(df_c_f, name)
    p_bear, p_base, p_bull = ScenarioPredictNextMonth(df_c_f["y_ret_next"], recent_window=36)
    print(f"{name} scenarios (next month): Bear={p_bear:.1%}  Base={p_base:.1%}  Bull={p_bull:.1%}")
    print()



formatA = "%Y-%m-%d"
formatB = "%m/%d/%y"



FullFlow("../data/DIA.csv", formatA)
FullFlow("../data/QQQ.csv", formatA)
FullFlow("../data/SPY.csv",formatB)

# dia["Date"] = pd.to_datetime(dia["Date"], format="%Y-%m-%d")
# qqq["Date"] = pd.to_datetime(qqq["Date"], format="%Y-%m-%d")
# spy["Date"] = pd.to_datetime(spy["Date"], format="%m/%d/%y")
    
# dia_clean = CleanData(dia, "DIA")
# qqq_clean = CleanData(qqq, "QQQ")
# spy_clean = CleanData(spy, "SPY")

# dia_c_f = AddFeatures(dia_clean)
# qqq_c_f = AddFeatures(qqq_clean)
# spy_c_f = AddFeatures(spy_clean)

# qqq_c_f = AddYUp(qqq_c_f)

# folds = GenerateFolds(qqq_c_f)

# pred_rows_list = []

# CombineFolds(qqq_c_f,folds)

# interpret_predictions(pred_rows_list)

# PredictNextMonth(qqq_c_f)

# p_bear, p_base, p_bull = ScenarioPredictNextMonth(qqq_c_f["y_ret_next"], recent_window=36)
# print(f"QQQ scenarios (next month): Bear={p_bear:.1%}  Base={p_base:.1%}  Bull={p_bull:.1%}")

