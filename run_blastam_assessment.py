# %%
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gzip

# 全域 DEBUG 參數，若為 True 則印出 debug 訊息
DEBUG = True

def parse_datetime_custom(x):
    """
    將原始的日期字串轉換成 datetime 物件，
    若包含 "24:00:00" 則轉換為隔日 00:00:00。
    """
    try:
        if isinstance(x, pd.Timestamp):
            return x
        if "24:00:00" in x:
            # 例如 "2025/3/10 24:00:00" 改為 "2025/3/10 00:00:00" 並加一天
            base_str = x.replace("24:00:00", "00:00:00")
            dt = datetime.strptime(base_str, "%Y/%m/%d %H:%M:%S")
            dt += timedelta(days=1)
            return dt
        else:
            return datetime.strptime(x, "%Y/%m/%d %H:%M:%S")
    except Exception as e:
        if DEBUG:
            print(f"Error parsing datetime '{x}': {e}")
        return pd.NaT

def read_weather_data(base_dir, station_id, year, month, debug=False):
    """
    根據指定的 base_dir、站點ID、年份與月份讀取氣象資料檔案，
    並依據檔案內的 header 自動調整讀取起始位置。
    """
    file_path = os.path.join(base_dir, station_id, f"{year}-{month}.csv.gz")
    if debug:
        print(f"Reading file: {file_path}")
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        all_lines = f.readlines()
    header_index = None
    for i, line in enumerate(all_lines):
        if "年月日時" in line:
            header_index = i
            if debug:
                print(f"Found header row at line {i}: {line.strip()}")
            break
    if header_index is None:
        raise ValueError("No header row found in file")
    
    from io import StringIO
    csv_data = "".join(all_lines[header_index:])
    data = pd.read_csv(StringIO(csv_data), encoding='utf-8')
    
    if debug:
        print(f"Initial data shape: {data.shape}")
    
    data = data.dropna(subset=["年月日時"])
    
    data["年月日時"] = data["年月日時"].apply(parse_datetime_custom)
    
    if debug:
        print("Converted '年月日時' to datetime. Sample values:")
        print(data["年月日時"].head())
    
    return data

def load_weather_data(base_dir, station_id, start_date, end_date, debug=False):
    """
    根據指定站點與時間範圍，整合多個月份的資料。
    """
    data_frames = []
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        try:
            df = read_weather_data(base_dir, station_id, year, month, debug=debug)
            data_frames.append(df)
            if debug:
                print(f"Loaded data for {year}-{month}, shape: {df.shape}")
        except Exception as e:
            if debug:
                print(f"Error loading data for {station_id} {year}-{month}: {e}")
        current_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)
    if len(data_frames) == 0:
        raise ValueError(f"No data loaded for station {station_id}")
    weather_data = pd.concat(data_frames, ignore_index=True)
    if debug:
        print(f"Combined weather data shape for station {station_id}: {weather_data.shape}")
    return weather_data

def prepare_model_input(five_day_data):
    """
    將 5 日天氣資料整理為模型所需的各項 numpy 陣列。
    注意：由於原始檔案中可能有多個相同欄位，本函式以第一個欄位為準，
    並利用 pd.to_numeric() 進行轉換，避免非數值型態導致 np.isnan() 錯誤。
    """
    if DEBUG:
        print("Data types before conversion:")
        print(five_day_data[['気温(℃)', '風速(m/s)', '降水量(mm)', '日照時間(時間)']].dtypes)
    
    temp_series = pd.to_numeric(five_day_data['気温(℃)'], errors='coerce')
    wind_series = pd.to_numeric(five_day_data['風速(m/s)'], errors='coerce')
    rainfall_series = pd.to_numeric(five_day_data['降水量(mm)'], errors='coerce')
    sunshine_series = pd.to_numeric(five_day_data['日照時間(時間)'], errors='coerce').fillna(0)
    
    return temp_series.values, wind_series.values, rainfall_series.values, sunshine_series.values

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

def calculate_blast_risk(station_id, date, base_dir, debug=False):
    try:
        date = pd.to_datetime(date)
        start_date = date - pd.Timedelta(days=4)
        end_date = date.replace(hour=23)
        first_day_shift = timedelta(days=1) if start_date.day == 1 else timedelta(days=0)
        weather_data = load_weather_data(base_dir, station_id, start_date - first_day_shift, end_date, debug=debug)
        if debug:
            print(f"Weather data for station {station_id} between {start_date} and {end_date}:")
            print(weather_data.head(10))
        five_day_data = weather_data[(weather_data['年月日時'] >= start_date) & (weather_data['年月日時'] <= end_date)]
        if debug:
            print(f"Filtered five-day data shape: {five_day_data.shape}")
        if len(five_day_data) != 120:
            raise ValueError(f"Data length error, For {start_date} to {end_date} at station {station_id}, {len(five_day_data)} provided")
        temp_5d, wind_5d, rainfall_5d, sun_shine_5d = prepare_model_input(five_day_data)
        if np.isnan(temp_5d).any() or np.isnan(wind_5d).any() or np.isnan(rainfall_5d).any() or np.isnan(sun_shine_5d).any():
            if debug:
                print("NaN values found in model input data.")
            return None
        leaf_wet_dict, results = koshimizu_model(temp_5d, wind_5d, rainfall_5d, sun_shine_5d)
        if debug:
            print(f"Calculated results for station {station_id}: {results}")
        return results
    except Exception as e:
        print(f"Error processing station {station_id}: {e}")
        return None
# %%
def main():
    #base_dir = r"D:\AMeDAS_visualization\weather_data"
    base_dir = r"./weather_data_repo/weather_data"
    result_dir = 'data'
    os.makedirs(result_dir, exist_ok=True)
    
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(31)]
    global DEBUG
    DEBUG = False
    if DEBUG:
        dates = [(datetime.now() - timedelta(days=8) - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1)]
    
    for date in dates:
        results = []
        for station_id in os.listdir(base_dir):
            station_path = os.path.join(base_dir, station_id)
            if os.path.isdir(station_path):
                res = calculate_blast_risk(station_id, date, base_dir, debug=DEBUG)
                if res:
                    results.append([station_id, res['blast_score']])
            if DEBUG:
                break
        result_file = os.path.join(result_dir, f"{date}.csv")
        result_df = pd.DataFrame(results, columns=['Station ID', 'Blast Score'])
        result_df.to_csv(result_file, index=False)
        if DEBUG:
            break

if __name__ == "__main__":
    main()

# %%
