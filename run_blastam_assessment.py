# %%
import os
import pandas as pd
import numpy as np
import logging
import gzip
from datetime import datetime, timedelta

# logging 設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 全域 DEBUG 參數
DEBUG = False


def parse_datetime_custom(x):
    """
    將原始的日期字串轉換成 datetime，
    若有 '24:00:00' 則轉為隔日 00:00:00。
    """
    try:
        if isinstance(x, pd.Timestamp):
            return x
        if "24:00:00" in x:
            base = x.replace("24:00:00", "00:00:00")
            dt = datetime.strptime(base, "%Y/%m/%d %H:%M:%S") + timedelta(days=1)
            return dt
        return datetime.strptime(x, "%Y/%m/%d %H:%M:%S")
    except Exception as e:
        logger.error(f"解析時間失敗 '{x}': {e}")
        return pd.NaT


def read_weather_data(base_dir, station_id, year, month):
    """
    讀取單月氣象資料，自動定位 header。
    若找不到 header 或解析失敗，記錄錯誤及前五行。
    """
    file_path = os.path.join(base_dir, station_id, f"{year}-{month}.csv.gz")
    logger.info(f"嘗試讀取檔案: {file_path}")
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"無法開啟檔案 {file_path}: {e}")
        return None

    # 定位 header
    header_idx = None
    for idx, line in enumerate(lines[:50]):  # 只搜尋前50行
        if "年月日時" in line:
            header_idx = idx
            logger.debug(f"{file_path} Header 在第 {idx} 行: {line.strip()}")
            break

    if header_idx is None:
        logger.error(f"找不到 header '年月日時' in {file_path}，前五行如下:")
        for ln in lines[:5]:
            logger.error(ln.strip())
        return None

    # 組合 CSV
    from io import StringIO
    csv_text = "".join(lines[header_idx:])
    try:
        df = pd.read_csv(StringIO(csv_text), encoding='utf-8')
    except Exception as e:
        logger.error(f"pd.read_csv 失敗 {file_path}: {e}\n前五行:\n" + "\n".join(lines[header_idx:header_idx+5]))
        return None

    # 清理與轉換
df = df.dropna(subset=["年月日時"])
df["年月日時"] = df["年月日時"].astype(str).apply(parse_datetime_custom)
    
    logger.info(f"讀取完畢 {file_path}, 資料量: {df.shape}")
    return df


def load_weather_data(base_dir, station_id, start_date, end_date):
    """
    整合多月資料，回傳 concat 後的 DataFrame。
    """
    dfs = []
    dt = start_date.replace(day=1)
    while dt <= end_date:
        y, m = dt.year, dt.month
        df = read_weather_data(base_dir, station_id, y, m)
        if df is not None:
            dfs.append(df)
        dt = (dt + timedelta(days=32)).replace(day=1)

    if not dfs:
        logger.error(f"站點 {station_id} 在 {start_date} 至 {end_date} 期間無任何資料")
        return None

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"合併後 {station_id} 資料量: {combined.shape}")
    return combined


def prepare_model_input(df):
    """
    擷取溫度、風速、降雨、日照等欄位，轉成 numpy。
    """
    cols = ['気温(℃)', '風速(m/s)', '降水量(mm)', '日照時間(時間)']
    arrays = []
    for col in cols:
        arr = pd.to_numeric(df[col], errors='coerce')
        arrays.append(arr.values)
        if DEBUG:
            logger.debug(f"{col} NaN 數量: {np.isnan(arr.values).sum()}")
    return arrays

# 假設 koshimizu_model 定義保持不變


def calculate_blast_risk(station_id, date_str, base_dir):
    """
    計算指定日期的 5 天風險，回傳結果 dict 或 None。
    """
    try:
        date = pd.to_datetime(date_str)
        start = date - timedelta(days=4)
        end = date.replace(hour=23)
        first_day = start.replace(day=1)
        df = load_weather_data(base_dir, station_id, first_day, end)
        if df is None:
            return None

        mask = (df['年月日時'] >= start) & (df['年月日時'] <= end)
        sub = df.loc[mask]
        if sub.shape[0] != 120:
            logger.error(f"{station_id} {date_str} 資料長度 {sub.shape[0]} !=120，前五筆: {sub.head().to_dict('records')}")
            return None

        temp, wind, rain, sun = prepare_model_input(sub)
        if any(np.isnan(arr).any() for arr in [temp, wind, rain, sun]):
            logger.error(f"{station_id} {date_str} 模型輸入含 NaN，前五筆: temp={temp[:5]}, wind={wind[:5]}")
            return None

        leaf_wet, res = koshimizu_model(temp, wind, rain, sun)
        logger.info(f"{station_id} {date_str} blast_score={res['blast_score']}")
        return res
    except Exception as e:
        logger.error(f"處理 {station_id} {date_str} 時發生例外: {e}")
        return None


def main():
    base_dir = './weather_data_repo/weather_data'
    result_dir = 'data'
    os.makedirs(result_dir, exist_ok=True)

    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(31)]
    for date in dates:
        results = []
        for station in os.listdir(base_dir):
            path = os.path.join(base_dir, station)
            if not os.path.isdir(path):
                continue
            res = calculate_blast_risk(station, date, base_dir)
            if res is not None:
                results.append([station, res['blast_score']])
        out = pd.DataFrame(results, columns=['Station', 'BlastScore'])
        fn = os.path.join(result_dir, f"{date}.csv")
        out.to_csv(fn, index=False)
        logger.info(f"已寫入結果: {fn}")

if __name__ == '__main__':
    main()
