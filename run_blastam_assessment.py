# %%
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gzip
def koshimizu_model(temp_5d,wind_5d,rainfall_5d,sun_shine_5d):
    """
    All parameters are FIVE-day HOURLY (0000–2300) data in numpy.array()
    (Totally 24 hours * 5 days = 120 data points)
    e.g. 2020/12/01 00:00 – 2020/12/05 23:00

    Units: temp_5d (℃), wind_5d (m/s), rainfall_5d (mm), sun_shine_5d (0–1 hour)

    Example:
    temp_5d=np.array([25. , 25.3, 25. , 24.8, 24.8, 24.5, 24.9, 25.7, 26. , 26.8, 28. ,
       30.1, 30.5, 30.5, 26.8, 24.9, 24.5, 25.1, 25.4, 25.2, 24.7, 24.6,
       24.4, 24.1, 24.2, 24.1, 24.1, 23.6, 23.8, 23.6, 24.1, 24.9, 25.5,
       27.5, 27. , 25.9, 25.3, 25.4, 25.4, 26.2, 26.7, 26.8, 26.3, 24.9,
       24.5, 23.7, 24. , 23.8, 23.4, 23.6, 23.6, 23.3, 23.8, 23.4, 23.7,
       24.3, 24.2, 25.7, 27.7, 27.7, 28.2, 28. , 27.6, 24.9, 25.5, 25.7,
       25.5, 24.8, 24.1, 23.9, 23.7, 23.2, 22. , 22. , 21.7, 20.5, 19.3,
       18.6, 19.6, 20.8, 21. , 20.4, 20.8, 21.9, 21.6, 21. , 21.5, 21.3,
       19.7, 19.4, 19.6, 18.8, 18.9, 19.5, 19.5, 19.7, 19.9, 19.8, 19.6,
       19.8, 19.8, 19.7, 20. , 20.9, 22.1, 22.8, 24.8, 26.2, 26.5, 25.6,
       25.1, 24.7, 24.4, 23.7, 23.7, 23.5, 23.5, 23.5, 23.6, 23.5])

    wind_5d=np.array([0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 1.9, 2.8,
       3.4, 1. , 2.1, 1.5, 1.4, 0.9, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 1. ,
       2.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 1.5, 1.6, 1.3, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 1.9, 1. , 0.1, 0.3, 0.1, 0.1, 0.1, 0.5, 0.1, 0.4,
       0.3, 0.8, 1.3, 0.8, 0.8, 0.1, 0.1, 1.3, 1.2, 0.6, 3.6, 1.6, 0.3,
       0.4, 0.1, 0.8, 0.1, 2. , 1.4, 3.1, 3.4, 0.1, 0.6, 3.4, 4. , 2.6,
       0.9, 2.1, 2.5, 2.4, 2.9, 2.1, 0.5, 0.7, 0.1, 1.7, 1.7, 2.9, 0.1,
       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1. , 0.1, 0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1, 0.8, 3.2, 2. , 2.4, 0.1, 0.1, 0.3, 1. , 0.4,
       0.1, 0.1, 0.1])

    rainfall_5d=np.array([ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  2.5, 29. , 17.5,  0.5,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0.5,  3.6,  1.7,  0.4,  0.2,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  1. ,  0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  2.5,  0.6,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1.7,  0.9,  0. ,  0.1,  0.5,
        5.1,  0. ,  0. ,  0. ,  3.1,  4.4,  0. ,  0. ,  0.6,  0.3,  0. ,
        1.1,  2.2,  0.5,  3.1,  2.5,  1.8,  0.8,  0.1,  0. ,  0.2,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0.3,  0.3,  0.2,  0.2,  0.1,  0. ,  0. ])
    sun_shine_5d=np.array([0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.1, 0.9, 0.9,
       0.8, 0. , 0. , 0. , 0.2, 0.1, 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0.1, 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0.1, 0.2, 0.3, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0.6, 0. , 0.2, 0. , 0.1, 0. , 0.1,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 2.2, 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0.2, 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. ])


    Although only the temperature is historical data that requires five days, 
    it is because the meteorological data are retrived at the same time. 
    Therefore, even rainfall, sunshine hours, and wind speed have to be 
    imported for five days. 
    In fact, all the above three meteorological factors can be filled with 0 
    since 16:00 on the previous day without any impact.
    """

    
    #基準１  1600から0600(0700)まで
    """
    午後4時から翌朝6時までの間に1
    時間でも降⾬の記録があれぱ,葉⾯湿潤時間は⾬の
    記録時間の1時間前からはじまり,翌朝7時まで継
    続するとみなすo
    ただし,その継続については次の
    ような条件を設け,それが満されないときは以下に
    のベるそれぞれの時点で中断したとみなす


    If there is a record of rainfall (for even one hour) between 4:00 pm and 
    6:00 am at the next morning, the leaf wetness duration is considered to 
    start one hour before the recording time and continue until 7:00 the next 
    morning, unless following conditions happen
    """
    rainfall_1600_0700=rainfall_5d[88:104]
    sun_shine_1600_0700=sun_shine_5d[88:104]
    wind_1600_0700=wind_5d[88:104]
    hour=16
    leaf_wet=False
    leaf_wet_dict={}
    accumulate_sunshine=0
    key=0
    for rainfall, sunshine, wind in zip(rainfall_1600_0700,sun_shine_1600_0700,wind_1600_0700):
        if (key<15): #set index bounding
            if (rainfall_1600_0700[key+1]>0):  #葉⾯湿潤時間は⾬の記録時間の1時間前からはじまり
                leaf_wet=True

        if (rainfall_1600_0700[key]>0 and sun_shine_1600_0700[key]==0.1):
            sun_shine_1600_0700[key]=0 #降雨と⽈照時間が同じ時間に記録されているときは0.1の⽈照時間に限０とみなす
        
        accumulate_sunshine=accumulate_sunshine+sun_shine_1600_0700[key]

        if hour==0:
            accumulate_sunshine=0  #翌日リセット

        if (accumulate_sunshine>0.2):
            leaf_wet=False #合計⽇照時間が0.2以下ときだけ⽈没後に継続するものとする。

        if (wind>=4):
            leaf_wet=False #⾬があっても同じ時間の⾵遠が4m以上のときは,その時間からはじまる葉⾯湿潤時間を推定しないことにするo
        
        if (key>1 and key<15):  #set index bounding
            #葉面湿潤時間は午後4時から翌朝4時までは,連続3時間の平均風速が3m未満のときだけ継続し
            #1時間でも風速が4m以上のときはその直前で3m風速が3峙間以上続くときは、2時間続いたところで中断するものとする
            if ((wind_1600_0700[key-1]>=3 and wind_1600_0700[key]>=3 and wind_1600_0700[key+1]>=3) and (hour>=16 or hour<=4)):
                leaf_wet=False 

            #1時間でも風速が4m以上のときはその直前で3m風速が3峙間以上続くときは、2時間続いたところで中断するものとする    
            if ((wind_1600_0700[key+1]>=4) and (hour>=16 or hour<=4)):
                leaf_wet=False 
        
        #午前4時から午前7時までは1時間でも3m以上の⾵速があればその時間で中断とする。ただし,降⾬と同じ時間の3mの⾵速は2mとみなす。
        if (hour>=4 or hour<=7) and ((rainfall==0 and wind>=3) or (rainfall>0 and wind>=4)):
                leaf_wet=False   
        #debug
        #print ("{:2d} hour, rainfall: {}, sunshine:{}, wind:{}, leaf_wet: {}".format(hour,rainfall,sunshine,wind,leaf_wet))
        leaf_wet_dict[hour]=leaf_wet
        hour=hour+1
        hour=hour%24
        key=key+1
    #基準2 0600-1600
    """
    午前6時から午後4時までの降⾬についても降⾬時刻、⽇照時間及び⾵速の状況によつては
    基準lによつて推定される葉⾯湿潤時間のはじまりを午後4時により近づけ,そのおわりを
    午前7時以降に延⾧するために葉⾯湿潤時間の推定を⾏う。

    ただし,この基準によつて
    
    実際に葉⾯湿潤時間のはじまりが遡って延⾧されるのは,
    午後2ー3時と午後6ー7時,⼜は午後3ー4時と午後6ー8時に降⾬がある場合だけで,
    延⾧幅は2時間以上にはならない。
    
    ー⽅,葉⾯湿潤峙間のおわりの延⾧には降⾬時刻の
    制限はなく延⾧幅は1ー10数時間になる。

    午前 6 時以降午後 3 時までの間に降雨がある時、降雨時刻以後の 3 時間が、日照時間
    の合計 0.1 以下、各１時間の風速 2ｍ以下の条件で葉面湿潤時間（以下、湿潤時間という。）
    とみなす。
    """
    rainfall_0600_1600=rainfall_5d[102:113]
    sun_shine_0600_1600=sun_shine_5d[102:113]
    wind_0600_1600=wind_5d[102:113]
    hour=6
    leaf_wet=False
    accumulate_sunshine=0
    key=102
    leaf_wet_dict[8]=False
    leaf_wet_dict[9]=False
    leaf_wet_dict[10]=False
    leaf_wet_dict[11]=False
    leaf_wet_dict[12]=False
    leaf_wet_dict[13]=False
    leaf_wet_dict[14]=False
    leaf_wet_dict[15]=False
    for rainfall, sunshine, wind in zip(rainfall_0600_1600,sun_shine_0600_1600,wind_0600_1600):
        if (hour>7 and hour<16): #from 0800 to 1500
            #debug
            #print ("{:2d} hour, rainfall: {}, sunshine:{}, wind:{}, leaf_wet: {}".format(hour,rainfall,sunshine,wind,leaf_wet))
            if (rainfall>0):
                if (wind_5d[key-3]<3 and sun_shine_5d[key-3]<=0.1 and hour-3>7):
                    leaf_wet_dict[hour-3]=True    
                if (wind_5d[key-2]<3 and sun_shine_5d[key-2]<=0.1 and hour-2>7):
                    leaf_wet_dict[hour-2]=True           
                if (wind_5d[key-1]<3 and sun_shine_5d[key-1]<=0.1 and hour-1>7):
                    leaf_wet_dict[hour-1]=True           
                if (wind_5d[key-0]<3 and sun_shine_5d[key-0]<=0.1):
                    leaf_wet_dict[hour-0]=True      
                if (wind_5d[key+1]<3 and sun_shine_5d[key+1]<=0.1 and hour+1<16):
                    leaf_wet_dict[hour+1]=True      
                if (wind_5d[key+2]<3 and sun_shine_5d[key+2]<=0.1 and hour+2<16):
                    leaf_wet_dict[hour+2]=True      
                if (wind_5d[key+3]<3 and sun_shine_5d[key+3]<=0.1 and hour+3<16):
                    leaf_wet_dict[hour+3]=True                                                                                                                                            
        hour=hour+1
        hour=hour%24
        key=key+1

    #基準2の2 基準2の1によって推定された葉⾯湿潤時間が相互に一時間だけ間隔をおいて
    #その間の1時間の日照時間が０．１以下で風速も3m以下のときは,⼆つの葉⾯湿潤時間は連続するとみなす
    hour=6
    key=102
    for rainfall, sunshine, wind in zip(rainfall_0600_1600,sun_shine_0600_1600,wind_0600_1600):
        if (hour>7 and hour<16): #from 0800 to 1500
            if (leaf_wet_dict[hour]==False):
                if (leaf_wet_dict[hour-1]==True and leaf_wet_dict[hour+1]==True and sunshine<=0.1 and wind<=3):
                    leaf_wet_dict[hour]=True
        
        hour=hour+1
        hour=hour%24
        key=key+1



    #基準 5   l 時間 4mm以上, 3 mmでも２時間以上連続する降⾬があるとき
    #降⾬後の9時間,⼜は降⾬前9時間以内にはじまった葉⾯湿潤時間はいもち病の侵⼊に無効とみなす
    wind_1600_1500=wind_5d[88:112]
    rainfall_1600_1500=rainfall_5d[88:112]
    for hour in range(16,40):   
        if rainfall_1600_1500[hour-16]>4:
            for ineffective_hour in range(hour-9,hour+10):
                 if (ineffective_hour>=16 and ineffective_hour<=40):
                    hour_now=ineffective_hour % 24
                    leaf_wet_dict[hour_now] = -2
                    
    #The hour judged as invalid by Rule 5 is expressed as -2

    
    #Calculate start time and end time
    start=False
    end=False
    wet_period_hrs=0
    temp_avg=0
    temp_1600_1500=temp_5d[88:112]
    for hour in range(16,40):
        hour_now=hour % 24
        if (leaf_wet_dict[hour_now]==True):
            if (start==False):
                start=hour_now
            end=hour_now
            wet_period_hrs=wet_period_hrs+1
            temp_avg=temp_avg+temp_1600_1500[hour-16]
        elif (start!=False):
            break
    if (wet_period_hrs!=0):
        temp_avg=temp_avg/wet_period_hrs
    
    #「好適条件」とは、湿潤時間中の平均気温が 15～25℃、その継続時間が第 1 表の湿潤時
    #間以上で、直前 5 日間の平均気温が 20℃を超え、25℃未満の場合である。
    #「準好適条件」とは、湿潤時間が 10 時間以上であるが、
    #湿潤時間中の平均気温が 15～25℃の範囲内にないか、
    #直前 5 日間の平均気温が 20℃以下または 25℃以上である場合、
    #あるいは湿潤時間中の平均気温が 15～21℃であっても、その継続時間が第 1 表の湿潤時間
    #より若干小さい場合である。「好適条件なし」とは、湿潤時間が 10 時間未満の場合である。 
    
    temp_towetness_hour_lower_limit={15:17, 16:15, 17:14, 18:13, 19:12, 20:11, 21:10, 22:10, 23:10, 24:10, 25:10}
    temp_5d_mean=temp_5d.mean()
    
    #Scoring formula modified on 24/08/27
    blast_score=5
    if (wet_period_hrs<10): #「好適条件なし」とは、湿潤時間が 10 時間未満の場合である。
        blast_score=-1   
    else:            
        if (temp_avg>=15 and temp_avg<=25): #平均気温が 15～21℃であっても、その継続時間が第 1 表の湿潤時間より若干小さい場合である
            if (wet_period_hrs < temp_towetness_hour_lower_limit[round(temp_avg)]):
                blast_score=4
                
        if (temp_avg<15 or temp_avg>25): #湿潤時間中の平均気温が 15～25℃の範囲内にないか
            blast_score=3
        
        if (temp_5d_mean>25): #直前 5 日間の平均気温が 20℃以下または 25℃以上である場合
            blast_score=2
        if (temp_5d_mean<20): #直前 5 日間の平均気温が 20℃以下または 25℃以上である場合
            blast_score=1




    return leaf_wet_dict,{'start':start, 'end': end, 'wet_period_hrs':wet_period_hrs,'wet_avg_temp':temp_avg, 'blast_score':blast_score}
import numpy as np
import pandas as pd
import gzip
import os
from datetime import datetime, timedelta

def read_weather_data(station_id, year, month):
    """
    Reads the weather data for a given station and month.
    """
    file_path = f'weather_data_repo/weather_data/{station_id}/{year}-{month}.csv.gz'
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = pd.read_csv(f, skiprows=3, parse_dates=['年月日時'])
    return data

def load_weather_data(station_id, start_date, end_date):
    """
    Loads weather data from the relevant months.
    """
    data_frames = []
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        data_frames.append(read_weather_data(station_id, year, month))
        current_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)

    weather_data = pd.concat(data_frames)
    return weather_data

def prepare_model_input(five_day_data):
    """
    Prepares the input for the koshimizu_model from the five-day weather data.
    """
    temp_5d = five_day_data['気温(℃)'].values
    wind_5d = five_day_data['風速(m/s)'].values
    rainfall_5d = five_day_data['降水量(mm)'].values
    sun_shine_5d = five_day_data['日照時間(時間)'].fillna(0).values
    return temp_5d, wind_5d, rainfall_5d, sun_shine_5d

def calculate_blast_risk(station_id, date):
    try:
        date = pd.to_datetime(date)
        start_date = date - pd.Timedelta(days=4)
        end_date = date.replace(hour=23)
        #因為該月的第一天的00時的資料會在上一個月的檔案內，所以如果start_date是該月的第一天，要把start_date往前推一天
        first_day_shift = timedelta(days=1) if start_date.day == 1 else timedelta(days=0)
        weather_data = load_weather_data(station_id, start_date - first_day_shift, end_date)
        five_day_data = weather_data[(weather_data['年月日時'] >= start_date) & (weather_data['年月日時'] <= end_date)]

        if len(five_day_data) != 120:
            #print(five_day_data['年月日時'])
            for d in five_day_data['年月日時']:
                print(d)
            raise ValueError(f"Data length error, For {start_date} to {end_date} at station {station_id}, {len(five_day_data)} provided")
        #如果temp_5d, wind_5d, rainfall_5d, sun_shine_5d中有任何直是nan的話，就不做計算
        if np.isnan(temp_5d).any() or np.isnan(wind_5d).any() or np.isnan(rainfall_5d).any() or np.isnan(sun_shine_5d).any():
            print(f"{temp_5d}, {wind_5d}, {rainfall_5d}, {sun_shine_5d}")
            return None
        temp_5d, wind_5d, rainfall_5d, sun_shine_5d = prepare_model_input(five_day_data)
        leaf_wet_dict, results = koshimizu_model(temp_5d, wind_5d, rainfall_5d, sun_shine_5d)
        return results
    except Exception as e:
        print(f"Error processing station {station_id}: {e}")
        return None

def main():
    stations_dir = 'weather_data_repo/weather_data'
    result_dir = 'data'
    os.makedirs(result_dir, exist_ok=True)
    today = datetime.now().strftime('%Y-%m-%d')
    #Modify prediction length here
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(3)]
    DEBUG = False
    if DEBUG:
        dates = [(datetime.now() - timedelta(days=8) - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1)]
    for date in dates:
        results = []
        for station_id in os.listdir(stations_dir):
            if os.path.isdir(os.path.join(stations_dir, station_id)):
                result = calculate_blast_risk(station_id, date)
                if result:
                    results.append([station_id, result['blast_score']])
            if DEBUG:
                break
        result_file = os.path.join(result_dir, f"{date}.csv")
        result_df = pd.DataFrame(results, columns=['Station ID', 'Blast Score'])
        result_df.to_csv(result_file, index=False)
        if DEBUG:
            break

# %%
if __name__ == "__main__":
    main()


# %%
