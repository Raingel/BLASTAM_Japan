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

# 儲存嘗試解碼的編碼清單
ENCODINGS_TO_TRY = ['cp932', 'utf-8', 'shift_jis', 'euc_jp']


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
    """
    file_path = os.path.join(base_dir, station_id, f"{year}-{month}.csv.gz")
    logger.info(f"嘗試讀取檔案: {file_path}")
    try:
        raw = gzip.open(file_path, 'rb').read()
    except Exception as e:
        logger.error(f"無法開啟檔案 {file_path}: {e}")
        return None

    text = None
    for enc in ENCODINGS_TO_TRY:
        try:
            text = raw.decode(enc)
            logger.info(f"檔案 {file_path} 解碼成功，使用編碼：{enc}")
            break
        except Exception:
            logger.debug(f"編碼 {enc} 解碼失敗 {file_path}")
    if text is None:
        text = raw.decode('cp932', errors='ignore')
        logger.warning(f"強制使用 cp932 (忽略錯誤)：{file_path}")

    lines = text.splitlines(keepends=True)
    header_idx = next((i for i, ln in enumerate(lines[:50]) if "年月日時" in ln), None)
    if header_idx is None:
        logger.error(f"找不到 header '年月日時' in {file_path}")
        return None

    from io import StringIO
    csv_text = "".join(lines[header_idx:])
    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception as e:
        logger.error(f"pd.read_csv 失敗 {file_path}: {e}")
        return None

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
        df = read_weather_data(base_dir, station_id, dt.year, dt.month)
        if df is not None:
            dfs.append(df)
        dt = (dt + timedelta(days=32)).replace(day=1)
    if not dfs:
        logger.error(f"站點 {station_id} 在 {start_date} 至 {end_date} 期間無資料")
        return None
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"合併後 {station_id} 資料量: {combined.shape}")
    return combined


def prepare_model_input(df):
    """
    把傳進來的 df 四欄都轉成 numpy array，並補 0（保留舊邏輯）。
    """
    cols = ['気温(℃)', '風速(m/s)', '降水量(mm)', '日照時間(時間)']
    arrays = []
    for col in cols:
        arr = pd.to_numeric(df[col], errors='coerce')
        # 補 0
        arr.fillna(0, inplace=True)
        arrays.append(arr.values)
        if DEBUG:
            logger.debug(f"{col} NaN 數量 (prepare 後): {np.isnan(arr).sum()}")
    return arrays


def koshimizu_model(temp_5d, wind_5d, rainfall_5d, sun_shine_5d):
    """
    模型邏輯：計算葉面濕潤狀態及病害風險分數（公式略）。
    此處假設輸入皆為連續 5 日（120 小時）的資料。
    """
    rainfall_1600_0700 = rainfall_5d[88:104]
    sun_shine_1600_0700 = sun_shine_5d[88:104]
    wind_1600_0700 = wind_5d[88:104]
    hour = 16
    leaf_wet = False
    leaf_wet_dict = {}
    accumulate_sunshine = 0
    key = 0
    for rainfall, sunshine, wind in zip(rainfall_1600_0700, sun_shine_1600_0700, wind_1600_0700):
        if key < 15:
            if rainfall_1600_0700[key+1] > 0:
                leaf_wet = True
        if rainfall_1600_0700[key] > 0 and sun_shine_1600_0700[key] == 0.1:
            sun_shine_1600_0700[key] = 0
        accumulate_sunshine += sun_shine_1600_0700[key]
        if hour == 0:
            accumulate_sunshine = 0
        if accumulate_sunshine > 0.2:
            leaf_wet = False
        if wind >= 4:
            leaf_wet = False
        if key > 1 and key < 15:
            if ((wind_1600_0700[key-1] >= 3 and wind_1600_0700[key] >= 3 and wind_1600_0700[key+1] >= 3)
                and (hour >= 16 or hour <= 4)):
                leaf_wet = False 
            if (wind_1600_0700[key+1] >= 4) and (hour >= 16 or hour <= 4):
                leaf_wet = False 
        if (hour >= 4 or hour <= 7) and ((rainfall == 0 and wind >= 3) or (rainfall > 0 and wind >= 4)):
            leaf_wet = False   
        leaf_wet_dict[hour] = leaf_wet
        hour = (hour + 1) % 24
        key += 1

    rainfall_0600_1600 = rainfall_5d[102:113]
    sun_shine_0600_1600 = sun_shine_5d[102:113]
    wind_0600_1600 = wind_5d[102:113]
    hour = 6
    key = 102
    for h in range(8, 16):
        leaf_wet_dict[h] = False
    for rainfall, sunshine, wind in zip(rainfall_0600_1600, sun_shine_0600_1600, wind_0600_1600):
        if hour > 7 and hour < 16:
            if rainfall > 0:
                if wind_5d[key-3] < 3 and sun_shine_5d[key-3] <= 0.1 and hour-3 > 7:
                    leaf_wet_dict[hour-3] = True    
                if wind_5d[key-2] < 3 and sun_shine_5d[key-2] <= 0.1 and hour-2 > 7:
                    leaf_wet_dict[hour-2] = True           
                if wind_5d[key-1] < 3 and sun_shine_5d[key-1] <= 0.1 and hour-1 > 7:
                    leaf_wet_dict[hour-1] = True           
                if wind_5d[key-0] < 3 and sun_shine_5d[key-0] <= 0.1:
                    leaf_wet_dict[hour-0] = True      
                if wind_5d[key+1] < 3 and sun_shine_5d[key+1] <= 0.1 and hour+1 < 16:
                    leaf_wet_dict[hour+1] = True      
                if wind_5d[key+2] < 3 and sun_shine_5d[key+2] <= 0.1 and hour+2 < 16:
                    leaf_wet_dict[hour+2] = True      
                if wind_5d[key+3] < 3 and sun_shine_5d[key+3] <= 0.1 and hour+3 < 16:
                    leaf_wet_dict[hour+3] = True
        hour = (hour + 1) % 24
        key += 1

    hour = 6
    key = 102
    for rainfall, sunshine, wind in zip(rainfall_0600_1600, sun_shine_0600_1600, wind_0600_1600):
        if hour > 7 and hour < 16:
            if leaf_wet_dict[hour] == False:
                if (leaf_wet_dict[hour-1] == True and leaf_wet_dict[hour+1] == True 
                    and sunshine <= 0.1 and wind <= 3):
                    leaf_wet_dict[hour] = True
        hour = (hour + 1) % 24
        key += 1

    wind_1600_1500 = wind_5d[88:112]
    rainfall_1600_1500 = rainfall_5d[88:112]
    for hr in range(16, 40):   
        if rainfall_1600_1500[hr-16] > 4:
            for ineffective_hour in range(hr-9, hr+10):
                if ineffective_hour >= 16 and ineffective_hour <= 40:
                    hour_now = ineffective_hour % 24
                    leaf_wet_dict[hour_now] = -2

    start = False
    end = False
    wet_period_hrs = 0
    temp_avg = 0
    temp_1600_1500 = temp_5d[88:112]
    for hr in range(16, 40):
        hour_now = hr % 24
        if leaf_wet_dict.get(hour_now) == True:
            if not start:
                start = hour_now
            end = hour_now
            wet_period_hrs += 1
            temp_avg += temp_1600_1500[hr-16]
        elif start:
            break
    if wet_period_hrs != 0:
        temp_avg = temp_avg / wet_period_hrs

    temp_towetness_hour_lower_limit = {15:17, 16:15, 17:14, 18:13, 19:12, 20:11, 21:10, 22:10, 23:10, 24:10, 25:10}
    temp_5d_mean = temp_5d.mean()
    blast_score = 5
    if wet_period_hrs < 10:
        blast_score = -1   
    else:            
        if 15 <= temp_avg <= 25:
            if wet_period_hrs < temp_towetness_hour_lower_limit[round(temp_avg)]:
                blast_score = 4
        if temp_avg < 15 or temp_avg > 25:
            blast_score = 3
        if temp_5d_mean > 25:
            blast_score = 2
        if temp_5d_mean < 20:
            blast_score = 1

    return leaf_wet_dict, {'start': start, 'end': end, 'wet_period_hrs': wet_period_hrs,
                             'wet_avg_temp': temp_avg, 'blast_score': blast_score}




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

        sub = df[(df['年月日時'] >= start) & (df['年月日時'] <= end)].copy()
        logger.debug(f"{station_id} {date_str} 篩出 {sub.shape[0]} 筆")

        # 如果資料筆數不對，直接放棄
        if sub.shape[0] != 120:
            logger.error(f"{station_id} {date_str} 資料長度 {sub.shape[0]} != 120")
            return None

        # 先檢查其他三欄的 NaN 數量
        other_cols = ['気温(℃)', '風速(m/s)', '降水量(mm)']
        nan_counts = sub[other_cols].isna().sum()
        for col, cnt in nan_counts.items():
            if cnt > 20:
                logger.warning(f"{station_id} {date_str} 欄位 {col} 有 {cnt} 個 NaN，品質不足，跳過")
                return None

        # 只對日照時間做 fillna(0)
        sub['日照時間(時間)'] = sub['日照時間(時間)'].fillna(0)

        # 轉陣列並補 0（prepare_model_input 會對四欄都補）
        temp, wind, rain, sun = prepare_model_input(sub)

        # 最後跑模型
        _, res = koshimizu_model(temp, wind, rain, sun)
        return res

    except Exception as e:
        logger.error(f"處理 {station_id} {date_str} 時發生例外: {e}")
        return None


def main():
    base_dir = './weather_data_repo/weather_data'
    if DEBUG:
        base_dir = "D:/AMeDAS_visualization/weather_data"
    result_dir = 'data'
    os.makedirs(result_dir, exist_ok=True)

    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(31)]
    if DEBUG:
        dates = [(datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')]

    for date in dates:
        logger.info(f"開始評估: {date}")
        results = []
        for station in sorted(os.listdir(base_dir)):
            path = os.path.join(base_dir, station)
            if not os.path.isdir(path):
                continue
            res = calculate_blast_risk(station, date, base_dir)
            if res is not None:
                results.append([station, res['blast_score']])
        out = pd.DataFrame(results, columns=['Station', 'BlastScore'])
        fn = os.path.join(result_dir, f"{date}.csv")
        out.to_csv(fn, index=False)
        logger.info(f"{date} 完成，寫入 {len(results)} 筆")

if __name__ == '__main__':
    main()

# %%
