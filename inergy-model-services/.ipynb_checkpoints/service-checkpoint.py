# service.py
import numpy as np
import bentoml
import requests
#import scikit-learn
from bentoml import mlflow
from bentoml.io import PandasDataFrame
from bentoml.io import NumpyNdarray
from bentoml.io import Text
from bentoml.io import Multipart
from bentoml.exceptions import BentoMLException
from datetime import date
import pvlib
import pandas as pd
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import json
import pickle
from datetime import date
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.timeseries_generation import holidays_timeseries
import math
from datetime import timezone
from darts.dataprocessing.transformers import MissingValuesFiller
import holidays
from pytz import timezone
import pytz


def isholiday(x, holiday_list):
    if x in holiday_list:
        return True
    return False

def isweekend(x):
    if x == 6 or x == 0:
        return True
    return False

def create_calendar(timeseries, timestep_minutes, holiday_list, local_timezone):

    calendar = pd.DataFrame(
        timeseries.index.tolist(),
        columns=['datetime']
    )

    calendar['year'] = calendar['datetime'].apply(lambda x: x.year)
    calendar['month'] = calendar['datetime'].apply(lambda x: x.month)
    calendar['yearweek'] = calendar['datetime'].apply(
        lambda x: int(x.strftime("%V")) - 1)
    calendar['day'] = calendar['datetime'].apply(lambda x: x.day)
    calendar['hour'] = calendar['datetime'].apply(lambda x: x.hour)
    calendar['minute'] = calendar['datetime'].apply(lambda x: x.minute)
    calendar['second'] = calendar['datetime'].apply(lambda x: x.second)
    calendar['weekday'] = calendar['datetime'].apply(lambda x: x.weekday())
    calendar['dayofyear'] = calendar['datetime'].apply(lambda x: x.day_of_year)
    calendar['monthday'] = calendar['datetime'].apply(
        lambda x: int(x.strftime("%d")) - 1)
    calendar['weekend'] = calendar['weekday'].apply(lambda x: isweekend(x))
    calendar['yearday'] = calendar['datetime'].apply(
        lambda x: int(x.strftime("%j")) - 1)

    # first convert to utc and then to timestamp
    calendar['timestamp'] = calendar['datetime'].apply(lambda x: local_timezone.localize(
        x).replace(tzinfo=pytz.utc).timestamp()).astype(int)

    # national_holidays = Province(name="valladolid").national_holidays()
    # regional_holidays = Province(name="valladolid").regional_holidays()
    # local_holidays = Province(name="valladolid").local_holidays()
    # holiday_list = national_holidays + regional_holidays + local_holidays

    calendar['holiday'] = calendar['datetime'].apply(
        lambda x: isholiday(x.date(), holiday_list))
    WNweekday = calendar['datetime'].apply(
        lambda x: x.weekday() if not isholiday(x.date(), holiday_list) else 5 if x.weekday() == 4 else 6)
    calendar['WN'] = WNweekday + calendar['hour']/24 + calendar['minute']/(24*60)
    calendar['DN'] = calendar['hour'] + calendar['minute']/(60)
    return calendar


def add_cyclical_time_features(calendar):
    """
    The function below is useful to create sinusoidal transformations of time features.
    This article explains why 2 transformations are necessary:
    https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
    """

    calendar['month_sin'] = calendar['month'].apply(
        lambda x: math.sin(2*math.pi/12*(x-1)))
    calendar['weekday_sin'] = calendar['weekday'].apply(
        lambda x: math.sin(2*math.pi/7*(x)))
    calendar['monthday_sin'] = calendar['monthday'].apply(
        lambda x: math.sin(2*math.pi/30*(x)))
    calendar['yearday_sin'] = calendar['yearday'].apply(
        lambda x: math.sin(2*math.pi/365*(x)))
    calendar['hour_sin'] = calendar['hour'].apply(
        lambda x: math.sin(2*math.pi/24*(x)))
    calendar['yearday_sin'] = calendar['yearday'].apply(
        lambda x: math.sin(2*math.pi/52.1428*(x)))

    calendar['month_cos'] = calendar['month'].apply(
        lambda x: math.cos(2*math.pi/12*(x-1)))
    calendar['weekday_cos'] = calendar['weekday'].apply(
        lambda x: math.cos(2*math.pi/7*(x)))
    calendar['monthday_cos'] = calendar['monthday'].apply(
        lambda x: math.cos(2*math.pi/30*(x)))
    calendar['yearday_cos'] = calendar['yearday'].apply(
        lambda x: math.cos(2*math.pi/365*(x)))
    calendar['hour_cos'] = calendar['hour'].apply(
        lambda x: math.cos(2*math.pi/24*(x)))
    calendar['yearday_cos'] = calendar['yearday'].apply(
        lambda x: math.cos(2*math.pi/52.1428*(x)))

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 5, 1)
    calendar['month_sin'][:50000].plot()
    plt.subplot(2, 5, 2)
    calendar['weekday_sin'][:1000].plot()
    plt.subplot(2, 5, 3)
    calendar['monthday_sin'][:1000].plot()
    plt.subplot(2, 5, 4)
    calendar['yearday_sin'][:1000000].plot()
    plt.subplot(2, 5, 5)
    calendar['hour_sin'][:96].plot()

    plt.subplot(2, 5, 6)
    calendar['month_cos'][:50000].plot()
    plt.subplot(2, 5, 7)
    calendar['weekday_cos'][:1000].plot()
    plt.subplot(2, 5, 8)
    calendar['monthday_cos'][:1000].plot()
    plt.subplot(2, 5, 9)
    calendar['yearday_cos'][:1000000].plot()
    plt.subplot(2, 5, 10)
    calendar['hour_cos'][:96].plot()

    return calendar

def get_time_covariates(series, country_code='PT', id_name='0'):
    """ Do it the darts way"""

    if isinstance(series, pd.Series):
        series = darts.TimeSeries.from_series(series)

    year = datetime_attribute_timeseries(
        time_index=series, attribute='year')

    month = datetime_attribute_timeseries(
        time_index=series, attribute='month', cyclic=True)

    dayofyear = datetime_attribute_timeseries(
        time_index=series, attribute='dayofyear', cyclic=True)

    hour = datetime_attribute_timeseries(
        time_index=series, attribute='hour', cyclic=True)

    # minute = datetime_attribute_timeseries(
    #     time_index=series, attribute='minute', cyclic=True)

    dayofweek = datetime_attribute_timeseries(
        time_index=series, attribute='dayofweek', cyclic=True)

    weekofyear = datetime_attribute_timeseries(
        time_index=series, attribute='weekofyear', cyclic=True)

    # dayofyear = datetime_attribute_timeseries(
    #     time_index=series, attribute='dayofyear')

    holidays = holidays_timeseries(
        time_index=series.time_index, country_code=country_code)

    # weekofyear = darts.TimeSeries.from_series(
    #     series.time_index.isocalendar().week)

    ts_list_covariates =  year.stack(month). \
                               stack(dayofyear). \
                               stack(hour). \
                               stack(dayofweek). \
                               stack(weekofyear). \
                               stack(holidays)
    
    ts_list_covariates = [ts_list_covariates.univariate_component(i).pd_dataframe() for i in range(ts_list_covariates.n_components)]

    id_l_covariates = ["year", 
                            "month_sin",
                            "month_cos", 
                            "dayofyear_sin",
                            "dayofyear_cos",
                            "hour_sin", 
                            "hour_cos",
                            "dayofweek_sin", 
                            "dayofweek_cos",
                            "weekofyear_sin",
                            "weekofyear_cos",
                            "holidays"]
    ts_id_l_covariates = [id_name for _ in range(12)]

    return ts_list_covariates, id_l_covariates, ts_id_l_covariates

def impute(ts: pd.DataFrame,
           holidays,
           max_thr: int = -1,
           a: float = 0.3,
           wncutoff: float = 0.000694,
           ycutoff: int = 3,
           ydcutoff: int = 30,
           resolution: str = "15",
           debug: bool = False,
           name: str = "PT",
           l_interpolation: bool = False,
           cut_date_val: str = "20221208",
           min_non_nan_interval: int = 24):
    """
    Reads the input dataframe and imputes the timeseries using a weighted average of historical data
    and simple interpolation. The weights of each method are exponentially dependent on the distance
    to the nearest non NaN value. More specifficaly, with increasing distance, the weight of
    simple interpolation decreases, and the weight of the historical data increases. If there is
    a consecutive subseries of NaNs longer than max_thr, then it is not imputed and returned with NaN
    values.

    Parameters
    ----------
    ts
        The pandas.DataFrame to be processed
    holidays
        The holidays of the country this timeseries belongs to
    max_thr
        If there is a consecutive subseries of NaNs longer than max_thr,
        then it is not imputed and returned with NaN values. If -1, every 
        value will be imputed regardless of how long the consecutive 
        subseries of NaNs it belongs to is
    a
        The weight that shows how quickly simple interpolation's weight decreases as
        the distacne to the nearest non NaN value increases
    wncutoff
        Historical data will only take into account dates that have at most wncutoff distance
        from the current null value's WN(Week Number)
    ycutoff
        Historical data will only take into account dates that have at most ycutoff distance
        from the current null value's year
    ydcutoff
        Historical data will only take into account dates that have at most ydcutoff distance
        from the current null value's yearday
    resolution
        The resolution of the dataset
    debug
        If true it will print helpfull intermediate results
    l_interpolation
        Whether to only use linear interpolation 
    cut_date_val
        All dates before cut_date_val that have nan values are imputed using historical data
        from dates which are also before cut_date_val. Datetimes after cut_date_val are not affected
        by this
    min_non_nan_interval
        If after imputation there exist continuous intervals of non nan values that are smaller than min_non_nan_interval
        hours, these intervals are all replaced  by nan values

    Returns
    -------
    pandas.DataFrame
        The imputed dataframe
    """
    if max_thr == -1: max_thr = len(ts)
    if l_interpolation:
        imputed_values = ts[ts[ts.columns[0]].isnull()]

        #null_dates: Series with all null dates to be imputed
        null_dates = imputed_values.index

        if debug:
            for date in null_dates:
                print(date)

        #isnull: An array which stores whether each value is null or not
        isnull = ts[ts.columns[0]].isnull().values


        #d: List with distances to the nearest non null value
        d = [len(null_dates) for _ in range(len(null_dates))]

        #leave_nan: List with all the values to be left NaN because there are
        #more that max_thr consecutive ones
        leave_nan = [False for _ in range(len(null_dates))]

        #Calculating the distances to the nearest non null value that is earlier in the series
        count = 1
        for i in range(len(null_dates)):
            d[i] = min(d[i], count)
            if i < len(null_dates) - 1:
                if null_dates[i+1] == null_dates[i] + pd.offsets.DateOffset(minutes=int(resolution)):
                    count += 1
                else:
                    count = 1

        #Calculating the distances to the nearest non null value that is later in the series
        count = 1
        for i in range(len(null_dates)-1, -1, -1):
            d[i] = min(d[i], count)
            if i > 0:
                if null_dates[i-1] == null_dates[i] - pd.offsets.DateOffset(minutes=int(resolution)):
                    count += 1
                else:
                    count = 1

        #If d[i] >= max_thr // 2, that means we have a consecutive subseries of NaNs longer than max_thr.
        #We mark this subseries so that it does not get imputed
        for i in range(len(null_dates)):
            if d[i] == max_thr // 2:
                for ii in range(max(0, i - max_thr // 2 + 1), min(i + max_thr // 2, len(null_dates))):
                    leave_nan[ii] = True
            elif d[i] > max_thr // 2:
                leave_nan[i] = True

        #using max_thr for linear interp. for UC7
        res = ts.interpolate(inplace=False)

        null_zip = [(i, null_date) for (i, null_date) in enumerate(null_dates) if leave_nan[i]]

        for i, null_date in null_zip:
            res.loc[null_date] = np.NaN
        imputed_values = res[ts[ts.columns[0]].isnull()].copy()

    else:
        #Returning calendar of the country ts belongs to
        calendar = create_calendar(ts, int(resolution), holidays, timezone("UTC"))
        calendar.index = calendar["datetime"]
        imputed_values = ts[ts[ts.columns[0]].isnull()].copy()

        #null_dates: Series with all null dates to be imputed
        null_dates = imputed_values.index

        if debug:
            for date in null_dates:
                print(date)

        #isnull: An array which stores whether each value is null or not
        isnull = ts[ts.columns[0]].isnull().values

        #d: List with distances to the nearest non null value
        d = [len(null_dates) for _ in range(len(null_dates))]

        #leave_nan: List with all the values to be left NaN because there are
        #more that max_thr consecutive ones
        leave_nan = [False for _ in range(len(null_dates))]

        #Calculating the distances to the nearest non null value that is earlier in the series
        count = 1
        for i in range(len(null_dates)):
            d[i] = min(d[i], count)
            if i < len(null_dates) - 1:
                if null_dates[i+1] == null_dates[i] + pd.offsets.DateOffset(minutes=int(resolution)):
                    count += 1
                else:
                    count = 1

        #Calculating the distances to the nearest non null value that is later in the series
        count = 1
        for i in range(len(null_dates)-1, -1, -1):
            d[i] = min(d[i], count)
            if i > 0:
                if null_dates[i-1] == null_dates[i] - pd.offsets.DateOffset(minutes=int(resolution)):
                    count += 1
                else:
                    count = 1

        #If d[i] >= max_thr // 2, that means we have a consecutive subseries of NaNs longer than max_thr.
        #We mark this subseries so that it does not get imputed
        for i in range(len(null_dates)):
            if d[i] == max_thr // 2:
                for ii in range(max(0, i - max_thr // 2 + 1), min(i + max_thr // 2, len(null_dates))):
                    leave_nan[ii] = True
            elif d[i] > max_thr // 2:
                leave_nan[i] = True

        #This is the interpolated version of the time series
        ts_interpolatied = ts.interpolate(inplace=False)

        #We copy the time series so that we don't change it while iterating
        res = ts.copy()

        null_zip = [(i, null_date) for (i, null_date) in enumerate(null_dates) if not leave_nan[i]]

        for i, null_date in null_zip:

            #WN: Day of the week + hour/24 + minute/(24*60). Holidays are handled as
            #either Saturdays(if the real day is a Friday) or Sundays(in every other case)
            currWN = calendar.loc[null_date]['WN']
            currYN = calendar.loc[null_date]['yearday']
            currY = calendar.loc[null_date]['year']
            currDayOfYear = calendar.loc[null_date]['dayofyear']
            currH = calendar.loc[null_date]['hour']
            currDN = calendar.loc[null_date]['DN']

            #weight of interpolated series, decreases as distance to nearest known value increases
            w = np.e ** (-a * d[i])

            #Historical value is calculated as the mean of values that have at most wncutoff distance to the current null value's
            #WN, ycutoff distance to its year, and ydcutoff distance to its yearday
            #All dates before cut_date_val that have nan values are imputed using historical data
            #from dates which are also before cut_date_val
            dcutoff = 6
            while True:   
                if null_date < pd.Timestamp(cut_date_val):
                    historical = ts[(~isnull) & (ts[ts.columns[0]].index < pd.Timestamp(cut_date_val)) &\
                                        ((((calendar['yearday'] - currYN) < 0) &\
                                        ((-calendar['yearday'] + currYN) < dcutoff)) &\
                                        ((calendar['DN'] - currDN < int(resolution)/120) & (- calendar['DN'] + currDN < int(resolution)/120)))][ts.columns[0]]

                
                #Dates after cut_date_val are not affected by cut_date_val
                else:
                    historical = ts[(~isnull) & ((((calendar['yearday'] - currYN) < 0) &\
                                        ((-calendar['yearday'] + currYN) < dcutoff)) &\
                                        ((calendar['DN'] - currDN < int(resolution)/120) & (- calendar['DN'] + currDN < int(resolution)/120)))][ts.columns[0]]
            
                if historical.empty:
                    dcutoff += 1
                    if dcutoff>20: 
                        break
                    continue


                historical = historical.mean()

                #imputed value is calculated as a wheighted average of the histrorical value and the value from intrepolation
                res.loc[null_date] = w * ts_interpolatied.loc[null_date] + (1 - w) * historical

                imputed_values.loc[null_date] = res.loc[null_date]

                if debug:
                    print(res.loc[null_date])
                break
    non_nan_intervals_to_nan = {}
    if min_non_nan_interval != -1:
        #UC7 Do that for l_interpolation also
        #If after imputation there exist continuous intervals of non nan values in the train set that are smaller 
        #than min_non_nan_interval time steps, these intervals are all replaced by nan values
        not_nan_values = res[(~res[res.columns[0]].isnull())]
        not_nan_dates = not_nan_values.index
        prev = not_nan_dates[0]
        start = prev

        for not_nan_day in not_nan_dates[1:]:
            if (not_nan_day - prev)!= pd.Timedelta(int(resolution), "min"):
                if prev - start < pd.Timedelta(int(resolution) * min_non_nan_interval, "min"):
                    print(f"Non Nan interval from {start} to {prev} is smaller than {min_non_nan_interval} time steps. Making this also Nan")
                    for date in pd.date_range(start=start, end=prev, freq=resolution + "min"):
                        non_nan_intervals_to_nan[date] = res.loc[date].values[0]
                        res.loc[date] = pd.NA
                        imputed_values.loc[date] = pd.NA

                start = not_nan_day
            prev = not_nan_day
        if prev - start < pd.Timedelta(int(resolution) * min_non_nan_interval, "min"):
            for date in pd.date_range(start=start, end=prev, freq=resolution + "min"):
                non_nan_intervals_to_nan[date] = res.loc[date].values[0]
                res.loc[date] = pd.NA
                imputed_values.loc[date] = pd.NA
    imputed_values = imputed_values[(~imputed_values[imputed_values.columns[0]].isnull())]
    non_nan_intervals_to_nan = pd.DataFrame.from_dict(non_nan_intervals_to_nan, columns=[res.columns[0]], orient='index')
    non_nan_intervals_to_nan.index.name = "Datetime"
    if not res.empty:
        full_res = res.asfreq(str(resolution)+'min')
    else:
        full_res = res.copy()

    if not imputed_values.empty:
        full_imputed_values = imputed_values.asfreq(str(resolution)+'min')
    else:
        full_imputed_values = imputed_values.copy()
    
    if not non_nan_intervals_to_nan.empty:
        full_non_nan_intervals_to_nan = non_nan_intervals_to_nan.asfreq(str(resolution)+'min')
    else:
        full_non_nan_intervals_to_nan = non_nan_intervals_to_nan.copy()

    #plot(full_res, full_imputed_values, full_non_nan_intervals_to_nan, name)
    return res#, imputed_values


def get_weather_covariates(start, end, fields=["shortwave_radiation"], name="W6 positive_active", inference=False):
    if type(fields) == str:
        fields = [fields]
    if inference:
        req_fields = ",".join(fields)
        result = requests.get(
            f"https://api.open-meteo.com/v1/gfs?latitude=42.564&longitude=12.643&hourly={req_fields}&forecast_days=10&timezone=auto"
        ).text
        result = json.loads(result)
        df_list = []
        for field in fields:
            data = result['hourly'][field]
            index = pd.to_datetime(result['hourly']['time'])
            assert len(data) == len(index)
            df = pd.DataFrame(data=data, index=index, columns=[field]).asfreq("60min")
            df.index.name = "Datetime"
            df_list.append(df)
        if name in ["W6 positive_active", "W6 positive_reactive"]:
            result_db = requests.get(
                    f"http://38.242.137.200:8000/api/v1/query?source=gfs_forecast&coordinates=(42.569,%2012.608)&start_date={start}&end_date={end}&fields={req_fields}"
                ).text
        elif name in ["W4 positive_active", "W4 positive_reactive"]:
            result_db = requests.get(
                    f"http://38.242.137.200:8000/api/v1/query?source=gfs_forecast&coordinates=(42.567,%2012.607)&start_date={start}&end_date={end}&fields={req_fields}"
                ).text
        else:
            result_db = requests.get(
                f"http://38.242.137.200:8000/api/v1/query?source=gfs_forecastt&coordinates=(42.564,%2012.643)&start_date={start}&end_date={end}&fields={req_fields}"
            ).text
        result_db = json.loads(result_db)
        df_list_db = []
        for field in fields:
            data = result_db['results'][field]['value']
            index = pd.to_datetime(result_db['time'])
            assert len(data) == len(index)
            df = pd.DataFrame(data=data, index=index, columns=[field]).asfreq("60min")
            df.index.name = "Datetime"
            end_date = '2023-10-28'

            # Specify the shift time    
            shift_time = 0

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[:end_date] = subset_df_shifted

            start_date = '2023-10-06'
            end_date = '2023-10-13'

            # Specify the shift time    
            shift_time = 3

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:end_date] = subset_df_shifted

            start_date = '2023-10-14'
            end_date = '2023-10-18'

            # Specify the shift time    
            shift_time = 2

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:end_date] = subset_df_shifted

            start_date = '2023-10-29'
            end_date = '2023-10-29'

            # Specify the shift time    
            shift_time = -4

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:end_date] = subset_df_shifted

            
            start_date = '2023-10-29'

            # Specify the shift time    
            shift_time = -1

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:] = subset_df_shifted


            # shift_time = -2

            # df = df.shift(periods=shift_time)
            df_list_db.append(df)
        result = []
        for ts, ts_db in zip(df_list, df_list_db):
            ts = ts[ts.index > ts_db.index[-1]]
            result_df = pd.concat([ts_db, ts])
            result.append(result_df)

        return result
    else:
        start = start.strftime("%Y-%m-%d")
        end = end.strftime("%Y-%m-%d")
        req_fields = ",".join(fields)
        if name in ["W6 positive_active", "W6 positive_reactive"]:
            result = requests.get(
                f"http://38.242.137.200:8000/api/v1/query?source=gfs_forecast&coordinates=(42.569,%2012.608)&start_date={start}&end_date={end}&fields={req_fields}"
            ).text
        elif name in ["W4 positive_active", "W4 positive_reactive"]:
            result = requests.get(
                f"http://38.242.137.200:8000/api/v1/query?source=gfs_forecast&coordinates=(42.567,%2012.607)&start_date={start}&end_date={end}&fields={req_fields}"
            ).text
        else:
            print("Error, no weather available for this time series")
        result = json.loads(result)
        df_list = []
        for field in fields:
            data = result['results'][field]['value']
            index = pd.to_datetime(result['time'])
            assert len(data) == len(index)
            df = pd.DataFrame(data=data, index=index, columns=[field]).asfreq("60min")
            df.index.name = "Datetime"

            end_date = '2023-10-28'

            # Specify the shift time    
            shift_time = 0

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[:end_date] = subset_df_shifted

            start_date = '2023-10-06'
            end_date = '2023-10-13'

            # Specify the shift time    
            shift_time = 3

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:end_date] = subset_df_shifted

            start_date = '2023-10-14'
            end_date = '2023-10-18'

            # Specify the shift time    
            shift_time = 2

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:end_date] = subset_df_shifted

            start_date = '2023-10-29'
            end_date = '2023-10-29'

            # Specify the shift time    
            shift_time = -4

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:end_date]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:end_date] = subset_df_shifted

            
            start_date = '2023-10-29'

            # Specify the shift time    
            shift_time = -1

            # Select the subset of the dataframe within the specified date range
            subset_df = df.loc[start_date:]

            # Shift the values forward by the specified time for the selected subset
            subset_df_shifted = subset_df.shift(periods=shift_time)

            # Update the original dataframe with the shifted values
            df.loc[start_date:] = subset_df_shifted


            # shift_time = -2

            # df = df.shift(periods=shift_time)

            df_list.append(df)
        return df_list

def add_weather_covariates(start, end, res_future, id_l_future_covs, ts_id_l_future_covs, ts_id_l, fields=["shortwave_radiation"], inference=False):
    other_covs = (res_future == [])
    for i, ts in list(enumerate(ts_id_l)):
        if not inference:
            covs_nans = get_weather_covariates(start, end, fields, ts_id_l[i][0], inference)
        else:
            covs_nans = get_weather_covariates(start, 
                                               pd.Timestamp(date.today()).ceil(freq='D') + pd.Timedelta("10D"), 
                                               fields,
                                               ts_id_l[i][0],
                                               inference)
        covs = []
        for cov in covs_nans:
            covs.append(cov)
        if other_covs:
            res_future.append(covs) 
            id_l_future_covs.append(list(map(lambda elem : elem + "_" + ts_id_l[i][0], fields)))
            ts_id_l_future_covs.append([ts_id_l[i][0] for _ in range(len(covs))])
            #TODO check multivariate and multiple
            #TODO check past and future covs more exactly
        else:
            res_future[i].extend(covs) 
            id_l_future_covs[i].extend(list(map(lambda elem : elem + "_" + ts_id_l[i][0], fields)))
            ts_id_l_future_covs[i].extend([ts_id_l[i][0] for _ in range(len(covs))])
    return res_future, id_l_future_covs, ts_id_l_future_covs


def get_pv_forecast(ts_id, start=None, end=None, inference=False, kW=185, use_saved=False):
    # start = ts[0].index[0] - pd.Timedelta("1d"),    
    # end = ts[0].index[-1] + pd.Timedelta("1d"),
    #TODO Add parameters to mlflow
    #TODO Add visualization
    if use_saved:
        covs_list, cov_id_l, cov_ts_id_l = multiple_ts_file_to_dfs("/new_vol_300/opt/energy-forecasting-theo/uc7-data-ops/UC6_future_covs_aligned_utc_full.csv", True, "60")
        covs_list = covs_list[0]
        covs_list_final = [covs_list]
        #[['diffuse_radiation_W4 positive_active', 'direct_normal_irradiance_W4 positive_active', 'shortwave_radiation_W4 positive_active', 'temperature_W4 positive_active', 'windspeed_10m_W4 positive_active']]
        #temp_air, wind_speed, ghi, dhi, dni

        covs_list_final = []
        covs_list_final.append(covs_list[3])
        covs_list_final.append(covs_list[4])
        covs_list_final.append(covs_list[2])
        covs_list_final.append(covs_list[0])
        covs_list_final.append(covs_list[1])
        covs_list_final = [covs_list_final]
        print(f"Using covs of {cov_id_l[0][0]}")
        print(cov_id_l)
        return darts.TimeSeries.from_dataframe(pvlib_forecast(covs_weather=covs_list_final[0], start=start, end=end, kW=kW))
    else:
        covs_list, cov_id_l, cov_ts_id_l = add_weather_covariates(start-pd.Timedelta("1d"),
                                                              end+pd.Timedelta("1d"),
                                                              [], 
                                                              [],
                                                              [], 
                                                              [ts_id],
                                                              fields=["temperature","windspeed_10m","shortwave_radiation","diffuse_radiation","direct_normal_irradiance"],
                                                              inference=inference)
        

        covs_list_final = []
        for ts, id_t in zip(covs_list, cov_id_l):
            new = []
            for comp, id in zip(ts, id_t):
                temp = impute(comp,
                        holidays.IT(),
                        max_thr = 2000,
                        a = 0.3,
                        wncutoff = 0.000694,
                        ycutoff = 3,
                        ydcutoff = 30,
                        resolution = "60",
                        debug = False,
                        name = id,
                        l_interpolation = False,
                        cut_date_val = "20231022",
                        min_non_nan_interval = -1)
                new.append(temp)
                covs_list_final.append(new)
        print(f"Using covs of {ts_id[0]}")
        return pvlib_forecast(covs_weather=covs_list_final[0], start=start, end=end, kW=kW)

def pvlib_forecast(covs_weather=[], start=None, end=None, kW=185):
    #init params
    latitude=42.567055
    longitude=12.607027
    surface_tilt=0 
    surface_azimuth=180
    modules_per_string=25
    strings_per_inverter=215
    altitude=0
    location=Location(latitude, longitude, altitude=altitude)

    #make weather
    weather = covs_weather[3].copy()
    weather.columns = ["dhi"]
    weather["dni"] = covs_weather[4]
    weather["ghi"] = covs_weather[2]
    weather["temp_air"] = covs_weather[0]
    weather["wind_speed"] = covs_weather[1]

    if start:
        weather = weather.loc[(weather.index >= start)]

    if end:
        weather = weather.loc[(weather.index <= end)] 

    #initialize system
    module_name = 'Canadian_Solar_CS5P_220M___2009_'
    inverter_name = 'ABB__ULTRA_1100_TL_OUTD_2_US_690_x_y_z__690V_' #'Power_Electronics__FS3000CU15__690V_' #'ABB__PVS980_58_2000kVA_K__660V_' #'ABB__ULTRA_1100_TL_OUTD_2_US_690_x_y_z__690V_'

    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod') 
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    module = sandia_modules[module_name]
    inverter = sapm_inverters[inverter_name]
    temperature_model_parameters = {'a': -2.4, 'b': -0.0455, 'deltaT': 8}

    system=PVSystem(surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
                module_parameters=module, inverter_parameters=inverter,
                temperature_model_parameters=temperature_model_parameters,
                modules_per_string=modules_per_string, strings_per_inverter=strings_per_inverter
                )
    

    modelchain=ModelChain(system, location)

    modelchain.run_model(weather)
    solar_data=modelchain.results.ac
    solar_data=pd.DataFrame(solar_data, columns=(['Value']))

    # Convert to our kW
    solar_data = solar_data * kW / 1000000.0

    return solar_data


     
# Load the runners

#uc7
uc7_households_apiu_runner = bentoml.mlflow.get("uc7_households_apiu:latest").to_runner()
uc7_companies_apiu_runner = bentoml.mlflow.get("uc7_companies_apiu:latest").to_runner()
uc7_charging_stations_runner = bentoml.mlflow.get("uc7_charging_stations:latest").to_runner()
uc7_pool_apiu_runner = bentoml.mlflow.get("uc7_pool_apiu:latest").to_runner()

#uc6
uc6_w4_ac_runner = bentoml.mlflow.get("uc6_w4_ac:latest").to_runner()
uc6_w4_reac_runner = bentoml.mlflow.get("uc6_w4_reac:latest").to_runner()
uc6_w6_reac_runner = bentoml.mlflow.get("uc6_w6_reac:latest").to_runner()

#clustering
kmeans_runner = bentoml.mlflow.get("uc7_kmeans_euclidean_k14:5yqwuff4kgpq36qw").to_runner()

#uc2
lgbm_italy_runner = bentoml.mlflow.get("lgbm_italy:25isa3w4aoatp6qw").to_runner()
nbeats_portugal_runner = bentoml.mlflow.get("nbeats_portugal:dgbmdtg4asatp6qw").to_runner()

#create a service for all models of inergy
inergy_service = bentoml.Service("inergy_service", runners=[uc7_households_apiu_runner, 
                                  uc7_companies_apiu_runner, 
                                  uc7_charging_stations_runner, 
                                  uc7_pool_apiu_runner, 
                                  uc6_w4_ac_runner, 
                                  uc6_w4_reac_runner, 
                                  uc6_w6_reac_runner, 
                                  kmeans_runner, 
                                  lgbm_italy_runner, 
                                  nbeats_portugal_runner])

input_spec = Multipart(n=Text(), 
                       meter_id=Text(),
                       series=PandasDataFrame())
#input must be multipart, as it contains both strings and dataframes. Dataframes are converted from json
#to dataframe by bentoml during the api call
@inergy_service.api(input=input_spec, output=Text())
def predict_uc7(n : str, 
            meter_id: str,
            series : pd.DataFrame,
            ) -> pd.DataFrame:
    try:
        n = int(n)        
    except:
        raise BentoMLException("Can't convert parameters to int")
    
    model_input = {
        "multiple": "False",
        "weather_covariates":[],
        "resolution": "60",
        "ts_id_pred": meter_id,        
        "n": n,
        "history": series,
        "series_uri": "",
        "roll_size": "1",
        "future_covariates_uri": "None",
        "past_covariates_uri": "None",
        "batch_size": 1}
    #households
    if [meter_id] in [['Energy_Apiu_1_8_0_BBB6152.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6154.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6155.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6156.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6157.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6158.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6159.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6160.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6161.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6162.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6163.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6164.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6166.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6167.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6168.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6169.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6170.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6171.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6174.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6175.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6176.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6178.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6179.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6180.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6181.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6183.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6184.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6185.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6186.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6188.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6189.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6190.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6191.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6192.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6195.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6197.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6198.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6199.Wh']]:
        res = uc7_households_apiu_runner.predict.run([model_input])
    #companies
    if [meter_id] in [['Energy_Apiu_1_8_0_BBB6030.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6062.Wh'],
                    ['Energy_Apiu_1_8_0_BBB6065.Wh']]:
        res = uc7_companies_apiu_runner.predict.run([model_input])
    #charging stations
    if meter_id in ['Energy_Apiu_1_8_0_BBB6103.Wh']:
        res = uc7_charging_stations_runner.predict.run([model_input])
    #pool
    if meter_id in ['Energy_Apiu_1_8_0_BBB6028.Wh']:
        res = uc7_pool_apiu_runner.predict.run([model_input])
    #pv
    if meter_id in ['Energy_Ameno_2_8_0_BBB6052.Wh']:
        res = [get_pv_forecast(["W6 positive_active"], start=series.index[-1] + pd.Timedelta("1h"), end=series.index[-1] + pd.Timedelta(int(n), "h"), inference=True, kW=68, use_saved=False)]
    if meter_id in ['Energy_Ameno_2_8_0_BBB6062.Wh']:
        res = [get_pv_forecast(["W6 positive_active"], start=series.index[-1] + pd.Timedelta("1h"), end=series.index[-1] + pd.Timedelta(int(n), "h"), inference=True, kW=32, use_saved=False)]
    return res[0].to_json()


input_spec = Multipart(n=Text(), 
                       series_id=Text(),
                       series=PandasDataFrame())
#input must be multipart, as it contains both strings and dataframes. Dataframes are converted from json
#to dataframe by bentoml during the api call
@inergy_service.api(input=input_spec, output=Text())
def predict_uc6(n : str, 
            series_id: str,
            series : pd.DataFrame,
            ) -> pd.DataFrame:
    try:
        n = int(n)        
    except:
        raise BentoMLException("Can't convert parameters to int")
        
    if series_id == "W4 positive_active":
        ts = get_pv_forecast(["W6 positive_active"], start=series.index[0], end=series.index[-1] + pd.Timedelta(int(n), "h"), inference=True, kW=60, use_saved=False)
        input = ts[ts.index <= series.index[-1]]
        output = ts[ts.index > series.index[-1]]

        model_input = {
            "multiple": "False",
            "weather_covariates":[],
            "resolution": "60",
            "ts_id_pred": "W4 positive_active",        
            "n": n,
            "history": (series + input),
            "series_uri": "",
            "roll_size": "1",
            "future_covariates_uri": "None",
            "past_covariates_uri": "None",
            "batch_size": 1}
        res = uc6_w4_ac_runner.predict.run([model_input])
        return (res[0] - output).to_json()
    elif series_id == "W4 positive_reactive":
        model_input = {
            "multiple": "False",
            "weather_covariates":[],
            "resolution": "60",
            "ts_id_pred": "W4 positive_active",        
            "n": n,
            "history": series,
            "series_uri": "",
            "roll_size": "1",
            "future_covariates_uri": "None",
            "past_covariates_uri": "None",
            "batch_size": 1}
        res = uc6_w4_reac_runner.predict.run([model_input])
        return (res[0]).to_json()
    elif series_id == "W6 positive_active":
        return (-get_pv_forecast(["W6 positive_active"], start=series.index[-1]+ pd.Timedelta(int(1), "h"), end=series.index[-1] + pd.Timedelta(int(n), "h"), inference=True, kW=60, use_saved=False)).to_json()
    elif series_id == "W6 positive_reactive":
        model_input = {
            "multiple": "False",
            "weather_covariates":[],
            "resolution": "60",
            "ts_id_pred": "W4 positive_active",        
            "n": n,
            "history": series,
            "series_uri": "",
            "roll_size": "1",
            "future_covariates_uri": "None",
            "past_covariates_uri": "None",
            "batch_size": 1}
            
        res = uc6_w6_reac_runner.predict.run([model_input])
        return (res[0]).to_json()
    else:
        raise BentoMLException("series_id not supported")
        

location="./scaler.pkl"

# please uncomment in case you need to load new scaler from MLflow
# location = mlflow.artifacts.download_artifacts(
#     artifact_uri='s3://mlflow-bucket/29/6b6a5fc98fe04bfbb39064972e2865d1/artifacts/scaler.pkl', dst_path='./')

scaler = pickle.load(open(location, 'rb'))

@inergy_service.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict_clustering(input : np.ndarray) -> np.ndarray:
    input_scaled = scaler.transform(input)
    return kmeans_runner.predict.run(input_scaled)


input_spec = Multipart(n=Text(), 
                       series=PandasDataFrame())
#input must be multipart, as it contains both strings and dataframes. Dataframes are converted from json
#to dataframe by bentoml during the api call
@inergy_service.api(input=input_spec, output=Text())
def predict_lgbm_italy(n : str, 
            series : pd.DataFrame,
            ) -> pd.DataFrame:
    try:
        n = int(n)        
    except:
        raise BentoMLException("Can't convert parameters to int")
                    
    model_input = {
        "multiple": "False",
        "weather_covariates":[],
        "resolution": "60",
        "ts_id_pred": "None",        
        "n": n,
        "history": series,
        "series_uri": "",
        "roll_size": "24",
        "future_covariates_uri": "None",
        "past_covariates_uri": "None",
        "batch_size": 1}
    
    res = lgbm_italy_runner.predict.run([model_input])
    return str(res[0].to_dict())


input_spec = Multipart(n=Text(), 
                       series=PandasDataFrame())
#input must be multipart, as it contains both strings and dataframes. Dataframes are converted from json
#to dataframe by bentoml during the api call
@inergy_service.api(input=input_spec, output=Text())
def predict_nbeats_portugal(n : str, 
            series : pd.DataFrame,
            ) -> pd.DataFrame:
    try:
        n = int(n)        
    except:
        raise BentoMLException("Can't convert parameters to int")
                    
    model_input = {
        "multiple": "False",
        "weather_covariates":[],
        "resolution": "60",
        "ts_id_pred": "None",        
        "n": n,
        "history": series,
        "series_uri": "",
        "roll_size": "24",
        "future_covariates_uri": "None",
        "past_covariates_uri": "None",
        "batch_size": 1}
    
    res = nbeats_portugal_runner.predict.run([model_input])
    return str(res[0].to_dict())
