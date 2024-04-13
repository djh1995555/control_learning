#!/usr/bin/env python
from __future__ import print_function 
import json
import os
import re
import shutil
import sys
import datetime
import time
import yaml
import argparse
import psycopg2
import numpy as np
import pandas as pd
from pluspy import db_utils
import matplotlib.pyplot as plt
from subprocess import PIPE, STDOUT
from collections import OrderedDict
from utils.topics_and_signals import *
from datetime import datetime, timedelta
from utils.report_plotter import ReportPlotter
from utils.vehicle_dynamic_model import VehicleDynamicModel
from clickhouse_sqlalchemy import exceptions


SORT = False
TS = 'ts'
TRACTION = 'traction'
BENCHMARK = 'benchmark'
WEIGHT = 'weight'
TIRE_PRESSURE = 'tire_pressure'
TIRE_TEMPERATURE = 'tire_temperature'
WHEEL_SPEED_FRONT_LEFT = 'front_left'
WHEEL_SPEED_FRONT_RIGHT = 'front_right'
WHEEL_SPEED_REAR_LEFT = 'rear_left'
WHEEL_SPEED_REAR_RIGHT = 'rear_right'


RAIN = 0

DB_CONF = {
    "clickhouse": {
        "driver": "clickhouse+native",
        "port": 9000,
        "database": "bagdb",
        "host": "clickhouse-cn",
        "user": "plus_viewer",
        "password": "ex4u1balSAeR68uC",
    },
}
WEATHER_QUERY_SQL = (
"""
    SELECT
        vehicle,
        timestamp,
        weather
    FROM t_weather_info
    WHERE
        vehicle = '{}' AND
        timestamp >= '{}' AND
        timestamp < '{}'
        
"""
)
WEIGHT_QUERY_SQL = (
"""
    SELECT
        msg,
        start_time
    FROM 
        bag_events
    WHERE 
        category = 'Drive' and
        status = 0 and
        type = 2 and
        tag = 'LoadWeight' and
        vehicle='{}' and
        start_time >= '{}' and
        start_time < '{}'
"""
)
QUERY_DATA = [  TS,
                DBW_ENABLED,
                ACC_REPORT,
                PITCH_POSE,
                YAW_RATE,
                THROTTLE_OUTPUT,
                BRAKE_CMD,
                ENGINE_BRAKE,
                RETARDER,
                V_CURRENT_DBW,
                GEAR_RATIO,
                GEAR_REPORT,
                TOTAL_WEIGHT,
                ENGINE_FRICTION_TORQUE,
                STEERING_ANGLE,
                CLUTCH_SLIP
               ]
    
TARGET_DATA_1 = [ 
                ACC_REPORT,
                PITCH_POSE,
                YAW_RATE,
                THROTTLE_OUTPUT,
                V_CURRENT_DBW,
                GEAR_RATIO,
                ENGINE_FRICTION_TORQUE,
                STEERING_ANGLE,
                CLUTCH_SLIP,
                BENCHMARK
               ]
TARGET_DATA_2 = [ 
                ACC_REPORT,
                PITCH_POSE,
                YAW_RATE,
                TRACTION,
                V_CURRENT_DBW,
                STEERING_ANGLE,
                BENCHMARK
               ]

SQL_BAG_MSG = (
"""     SELECT ts,
    dbw_enabled,
    JSONExtractFloat(vehicle_control_cmd, 'debugCmd', 'aReport') as a_report,
    JSONExtractFloat(vehicle_status, 'posePitch') as pose_pitch,
    JSONExtractFloat(vehicle_control_cmd, 'debugCmd', 'yawrate') as yaw_rate,
    JSONExtractFloat(vehicle_dbw_reports, 'throttleReport', 'pedalOutput') as throttle_output,
    JSONExtractFloat(vehicle_control_cmd, 'brakeCmd', 'normalizedValue') as brake_cmd,
    JSONExtractFloat(vehicle_control_cmd, 'brakeCmd', 'engineBrakeTorquePct') as engine_brake,
    JSONExtractFloat(vehicle_control_cmd, 'brakeCmd', 'retarderTorquePct') as retarder,
    JSONExtractFloat(vehicle_dbw_reports, 'steeringReport', 'speed') as v_current_dbw,
    JSONExtractFloat(vehicle_dbw_reports, 'throttleInfoReport', 'throttleRate') as gear_ratio,
    JSONExtractFloat(vehicle_dbw_reports, 'gearReport', 'state') as gear,
    JSONExtractFloat(vehicle_control_cmd, 'debugCmd', 'totalWeight') as total_weight,
    JSONExtractFloat(vehicle_dbw_reports, 'throttleInfoReport', 'throttlePc') as throttle_pc,
    JSONExtractFloat(vehicle_dbw_reports, 'steeringReport', 'steeringWheelAngle') as steering_wheel_angle,
    JSONExtractFloat(vehicle_dbw_reports, 'gearInfoReport', 'clutchSlip') as clutch_slip
    FROM bagdb.bag_messages
    WHERE vehicle=:vehicle
    and ts between :start_ts and :end_ts
    ORDER by ts """
)


class DataCollector:
    def __init__(self,args):
        with open(args.config, 'r') as f:
            self._config = yaml.load(f)
        self._raw_data_by_vehicle = {}
        self._raw_data = pd.DataFrame(columns=QUERY_DATA)
        self._report_plotter = ReportPlotter('ReportGenerator')

        self._throttle_train_num = 0
        self._throttle_test_num = 0
        self._throttle_validation_num = 0
        
        self._traction_train_num = 0
        self._traction_test_num = 0
        self._traction_validation_num = 0
        
        self._weather_invalid_num = 0
        self._total_num = 0
        self._vehicle_dynamic_model = VehicleDynamicModel(self._config['vehicle_param'])
                   
    def data_query_from_database(self, vehicle_name, start_ts, end_ts):  
        temp_ts = start_ts
        query_delta_time = timedelta(minutes=self._config['query_step'])
        df = pd.DataFrame(columns=QUERY_DATA)
        empty_frame_num = 0
        while temp_ts + query_delta_time <= end_ts:
            with db_utils.db_session_open_close(DB_CONF['clickhouse']) as db_session:
                try:
                    df_new = db_session.execute(SQL_BAG_MSG,{"vehicle": vehicle_name,
                                                        "start_ts": temp_ts,
                                                        "end_ts": temp_ts + query_delta_time}).fetchall()                  
                except exceptions.DatabaseException as e:
                    continue
                
                if(len(df_new) != 0):
                    empty_frame_num = 0
                    new_df = pd.DataFrame(df_new,columns=QUERY_DATA)
                    df = pd.concat([df, new_df], ignore_index=True, sort=SORT)
                    # print("query interval {}~{}/{}: data length = {}".format(temp_ts + self._time_gap, 
                    #                                                    temp_ts + query_delta_time + self._time_gap, 
                    #                                                    end_ts + self._time_gap, len(df)))
                else:
                    empty_frame_num += 1
                    # print('no data queried!')
                    if(empty_frame_num > 10):
                        return df
                    
            temp_ts += query_delta_time
            
        if (temp_ts + query_delta_time > end_ts):
            with db_utils.db_session_open_close(DB_CONF['clickhouse']) as db_session:
                df_new = db_session.execute(SQL_BAG_MSG,{"vehicle": vehicle_name,
                                                    "start_ts": temp_ts,
                                                    "end_ts": end_ts}).fetchall()
                # print('df new len: {}'.format(len(df_new)))
                if(len(df_new) != 0):
                    df = pd.concat([df, pd.DataFrame(df_new, columns=QUERY_DATA)], ignore_index=True, sort=SORT)
                    # print("query interval {}~{}/{}: data length = {}".format(temp_ts + self._time_gap, 
                    #                                                         end_ts + self._time_gap, 
                    #                                                         end_ts + self._time_gap, len(df)))
                # else:
                #     print('no data queried!')

        # print("data query finish")
        return df

    def collect_bag_data(self, bag_data):
        raw_data_dict = {}
        for data_name in QUERY_DATA:
            if(data_name == 'ts'):
                break
            plot_data = bag_data[data_name]
            raw_data_dict[data_name] = np.array([x[1] for x in plot_data.data])
        raw_data = pd.DataFrame(raw_data_dict)
        return raw_data
    
    def extract_dict(self, dict):
        key, value = next(iter(dict.items()))
        return [key, value]
    
    def search_ground_truth(self, vehicle_name, time_interval):
        interval_start = datetime.strptime(time_interval[0], '%Y-%m-%d %H:%M:%S')
        interval_end = datetime.strptime(time_interval[1], '%Y-%m-%d %H:%M:%S')

        def extract_dict(dict):
            key, value = next(iter(dict.items()))
            return [key, value]        
        self._ground_truth = []
        ground_truth_dict = self._config['weight_ground_truth']
        ground_truth_by_time = ground_truth_dict[vehicle_name]
        
        left_ts = datetime.strptime(extract_dict(ground_truth_by_time[0])[0], '%Y-%m-%d %H:%M:%S')
        right_ts = datetime.strptime(extract_dict(ground_truth_by_time[-1])[0], '%Y-%m-%d %H:%M:%S')

        if(interval_start < left_ts - timedelta(days=1) or right_ts < interval_end):
            print('the search time interval exceed normal range!')
            return
        
        left_idx = 0
        while(left_ts < interval_start):
            left_idx += 1
            left_ts = datetime.strptime(extract_dict(ground_truth_by_time[left_idx])[0], '%Y-%m-%d %H:%M:%S')
            
        right_idx = left_idx
        right_ts = datetime.strptime(extract_dict(ground_truth_by_time[right_idx])[0], '%Y-%m-%d %H:%M:%S')
        while(right_ts < interval_end):
            right_idx += 1
            right_ts = datetime.strptime(extract_dict(ground_truth_by_time[right_idx])[0], '%Y-%m-%d %H:%M:%S')                    
        
        for i in range(left_idx, right_idx + 1):
            pair = extract_dict(ground_truth_by_time[i])
            time_stamp = datetime.strptime(pair[0], '%Y-%m-%d %H:%M:%S')
            self._ground_truth.append((time_stamp,pair[1]))

    def time_binary_search(self, data, target):
        left = 0
        right = len(data) - 1

        while(left <= right):
            mid = left + (right - left) / 2
            time = data[mid] + self._time_gap
            if(time < target):
                left = mid + 1
            elif (time > target):
                right = mid -1
            elif (time == target):
                return mid
        return left
    
    def add_ground_truth(self,df):
        total_length = len(df)
        ground_truth = [0] * total_length
        last_idx = 0
        for pair in self._ground_truth:
            idx = self.time_binary_search(df['ts'],pair[0])
            print('target:{}, idx:{}'.format(pair[0], idx))
            idx = min(idx,total_length)
            ground_truth[last_idx:idx] = [pair[1]] * (idx - last_idx)
            last_idx = idx
        df[WEIGHT] = ground_truth
        return df
    
    def plot_signal(self, output_dir, output_name, data_dict):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for segment_name, signals in data_dict.items():
            subplot_figure = None
            plot_html_str = ""
            figure_list = []
            # print('plot {}'.format(segment_name))
            for signal_name, data in signals.items():
                # print(' plot {}, size: {}, type: {}'.format(signal_name, len(data), type(data)))
                # print(' data: {}'.format(data))
                subplot = self._report_plotter.plot_figure_plotly(x_list = [np.array(range(len(data)))], 
                                                    y_list = [np.array(data)],
                                                    legend_list = ['{}'.format(signal_name)],
                                                    x_label = 'timestamp',
                                                    y_label = '{}'.format(signal_name),
                                                    title = '{}'.format(signal_name),
                                                    legend_prefix = '',
                                                    figure_height=self._config['signal_fig_height'],)
                figure_list.append(subplot)

            subplot_figure_list = [(i + 1, 1, fig) for i, fig in enumerate(figure_list)]
            subplot_figure = self._report_plotter.append_figure_to_subplot_plotly(subplot_figure_list, 
                                                                                row_num = len(figure_list), 
                                                                                col_num = 1, 
                                                                                template="plotly_dark", 
                                                                                subplot_fig=subplot_figure,
                                                                                figure_height = self._config['signal_fig_height'],
                                                                                vertical_spacing = self._config['vertical_spacig'])
            plot_html_str += self._report_plotter.get_fuel_fig_html_str({output_name: subplot_figure})
            html_str = self._report_plotter.generate_html_fuel_report(plot_html_str)
            with open(os.path.join(output_dir,'{}.html'.format(segment_name)), 'w') as f:
                f.write(html_str)
            
    def is_invalid_data(self, data):
        if(len(data[DBW_ENABLED]) == 0 ):
            return True
        return ((len(data[DBW_ENABLED]) > 0 and min(data[DBW_ENABLED]) == 0) or
                (len(data[BRAKE_CMD]) > 0 and max(data[BRAKE_CMD]) > 0) or
                (len(data[ENGINE_BRAKE]) > 0 and max(data[ENGINE_BRAKE]) > 0) or
                (len(data[RETARDER]) > 0 and max(data[RETARDER]) > 0) or
                (len(data[V_CURRENT_DBW]) > 0 and min(data[V_CURRENT_DBW]) < 10))  
        
    def orgnize_data(self, data):
        df = pd.DataFrame(columns=TARGET_DATA_1)
        df[ACC_REPORT] = data[ACC_REPORT]
        df[PITCH_POSE] = data[PITCH_POSE]
        df[YAW_RATE] = data[YAW_RATE]
        df[THROTTLE_OUTPUT] = data[THROTTLE_OUTPUT]
        df[V_CURRENT_DBW] = data[V_CURRENT_DBW]
        df[GEAR_RATIO] = data[GEAR_RATIO]
        df[ENGINE_FRICTION_TORQUE] = data[ENGINE_FRICTION_TORQUE]
        df[STEERING_ANGLE] = data[STEERING_ANGLE]
        df[CLUTCH_SLIP] = data[CLUTCH_SLIP]
        df[BENCHMARK] = self._vehicle_dynamic_model.compute_acc(data[THROTTLE_OUTPUT], 
                                                                data[GEAR_RATIO], 
                                                                data[ENGINE_FRICTION_TORQUE], 
                                                                data[V_CURRENT_DBW],
                                                                data[TOTAL_WEIGHT], 
                                                                data[PITCH_POSE])
        return df
    
    def orgnize_data_2(self, data):
        df = pd.DataFrame(columns=TARGET_DATA_2)
        if((len(data[CLUTCH_SLIP]) > 0 and max(data[CLUTCH_SLIP]) > 5)):
            return df
        df[ACC_REPORT] = data[ACC_REPORT]
        df[PITCH_POSE] = data[PITCH_POSE]
        df[YAW_RATE] = data[YAW_RATE]
        df[TRACTION] = self._vehicle_dynamic_model.compute_traction(data[THROTTLE_OUTPUT], 
                                                                    data[GEAR_RATIO], 
                                                                    data[ENGINE_FRICTION_TORQUE])
        df[V_CURRENT_DBW] = data[V_CURRENT_DBW]
        df[STEERING_ANGLE] = data[STEERING_ANGLE]
        df[BENCHMARK] = data[BENCHMARK]
        return df

    def deal_with_time_string(self, time_string):
        time_string = time_string.replace('-','_')
        time_string = time_string.replace(' ','_')
        time_string = time_string.replace(':','_')
        return time_string
        
    def get_segment_name(self, vehicle_name, start_ts, end_ts):
        start_time = self.deal_with_time_string('{}'.format(start_ts + self._time_gap))
        end_time = self.deal_with_time_string('{}'.format(end_ts + self._time_gap))
        return '{}_{}_to_{}'.format(vehicle_name, start_time, end_time) 

    def save_data(self, data_list, dir_name):
        # print('dir_name:{}, data num:{}'.format(dir_name, len(data_list)))
        dir = os.path.join(args.output_dir, dir_name) 
        
        train_data_dir = os.path.join(dir, 'train')
        if not os.path.exists(train_data_dir):
            os.makedirs(train_data_dir)
            
        validation_data_dir = os.path.join(dir, 'validation')
        if not os.path.exists(validation_data_dir):
            os.makedirs(validation_data_dir)
            
        test_data_dir = os.path.join(dir, 'test')
        if not os.path.exists(test_data_dir):
            os.makedirs(test_data_dir)
            
        data_num = len(data_list)
        train_data_range = int(data_num * self._config['train_percentage'])
        test_data_range = int(data_num * self._config['test_percentage'])
        train_data = data_list[0 : train_data_range]
        validation_data = data_list[train_data_range : data_num - test_data_range]
        test_data = data_list[data_num - test_data_range : -1]
        if(dir_name == 'orgnized_traction_data'):
            self._traction_train_num += len(train_data)
            self._traction_test_num += len(test_data)
            self._traction_validation_num += len(validation_data)
        else:
            self._throttle_train_num += len(train_data)
            self._throttle_test_num += len(test_data)
            self._throttle_validation_num += len(validation_data)           
        self.plot_and_save_csv(train_data, train_data_dir)
        self.plot_and_save_csv(validation_data, validation_data_dir)
        self.plot_and_save_csv(test_data, test_data_dir)
        
            
    def plot_and_save_csv(self, data_list, dir_name):
        # print('     save dir_name:{}, data num:{}'.format(dir_name, len(data_list)))
        data_dir = os.path.join(dir_name, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        label_dir = os.path.join(dir_name, 'label')
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
            
        plot_dir = os.path.join(dir_name, 'plot')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        signal_dict = OrderedDict()
        for name, label, data in data_list:
            # print('         name:{}'.format(name))
            data.to_csv(os.path.join(data_dir,'{}.csv'.format(name)), index=False)
            with open(os.path.join(label_dir,'{}.json'.format(name)), 'w') as f:
                json.dump(label, f)
            data_with_signals = {}
            for signal_name, signal_data in data.iteritems():
                data_with_signals[signal_name] = data[signal_name]
            signal_dict[name] = data_with_signals 
            
        self.plot_signal(plot_dir, '', signal_dict)         
        
    def save_signal(self, dict, data, segment_name, columns):
        data_with_signals = {}
        for signal in columns:
            data_with_signals[signal] = data[signal]
        dict[segment_name] = data_with_signals
            
    def data_process(self, 
                     vehicle_name, 
                     weight_gt, 
                     weather_info,
                     start_ts, 
                     end_ts,
                     original_data_list,
                     orgnized_throttle_data_list,
                     orgnized_traction_data_list):
        original_data = self.data_query_from_database(vehicle_name, start_ts, end_ts)
        segment_name = self.get_segment_name(vehicle_name, start_ts, end_ts)
        
        if(len(original_data)==0):
            return
    
        try:
            original_data = original_data.drop(TS, axis=1)
            original_data = original_data.interpolate(method='linear')
        except TypeError:
            return
        original_data = original_data.dropna()
        self._total_num += 1
        if(self.is_invalid_data(original_data)):
            # print('data is invalid!')
            return
        # print('data is valid!')
        
        label = {'vehicle': vehicle_name,
                 'start_ts': self.deal_with_time_string('{}'.format(start_ts + self._time_gap)),
                 'end_ts': self.deal_with_time_string('{}'.format(end_ts + self._time_gap)),
                 'mass': weight_gt,
                 'tire': 3,
                 'weather': weather_info[0][2]}

        original_data = original_data.iloc[::self._config['de_sample']]
        original_data_list.append((segment_name, label, original_data))
        
        orgnized_data = self.orgnize_data(original_data)
        orgnized_throttle_data_list.append((segment_name, label, orgnized_data))
        
        orgnized_data_2 = self.orgnize_data_2(orgnized_data)
        if(len(orgnized_data_2) > 0):
            orgnized_traction_data_list.append((segment_name, label, orgnized_data_2))
   
    def get_weight_gt(self, cursor, vehicle_name, date):
        start_of_day = datetime(date.year, date.month, date.day, 0, 0, 0)
        start_timestamp = float(time.mktime(start_of_day.timetuple()))
        end_of_day = datetime(date.year, date.month, date.day, 23, 59, 59)
        end_timestamp = float(time.mktime(end_of_day.timetuple()))

        cursor.execute(WEIGHT_QUERY_SQL.format(vehicle_name, start_timestamp, end_timestamp))
        records = cursor.fetchall()
        weight = ''
        if(len(records)!=0):
            weight = records[0][0]
        match = re.search(r"[-+]?\d*\.?\d+", weight)
        if match:
            weight = float(match.group())
        else:
            weight = 0 
        
        return weight
    
    def get_weather_info(self, cursor, vehicle_name, start_ts, end_ts):
        start_timestamp = float(time.mktime(start_ts.timetuple()))
        end_timestamp = float(time.mktime(end_ts.timetuple()))
        # print('========{}~{}========'.format(start_timestamp, end_timestamp))
        cursor.execute(WEATHER_QUERY_SQL.format(vehicle_name, start_timestamp, end_timestamp))
        records = cursor.fetchall()
        # print('records:{}'.format(len(records)))
        return records     
        
    def run(self):
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
        
        vehicle_names = np.array(self._config['vehciles']) 
        time_intervals = np.array(self._config['time_intervals'])
        query_delta_time = timedelta(minutes=self._config['query_delta_time'])
        save_delta_time = timedelta(hours=self._config['save_delta_time'])
        self._time_gap = timedelta(hours=8)
        self._total_minutes = 0
        
        weight_conn_string = "dbname='bagdb' host='sz-typostgres' user='bagdb_guest' password='guest@2022!'"
        weight_conn = psycopg2.connect(weight_conn_string)
        weight_cursor = weight_conn.cursor()  

        weather_conn_string = "host='labeling2' dbname='tractruck_new' user='tractruck_viewer' password='tractruck_viewer'"
        weather_conn = psycopg2.connect(weather_conn_string)
        weather_cursor = weather_conn.cursor()
        for vehicle_name in vehicle_names:
            for time_interval in time_intervals:
                start_ts = datetime.strptime(time_interval[0], '%Y-%m-%d %H:%M:%S')
                end_ts = datetime.strptime(time_interval[1], '%Y-%m-%d %H:%M:%S')
                self._total_minutes += (end_ts - start_ts).total_seconds() / 60
                
        start_time = datetime.now()        
        for vehicle_name in vehicle_names:
            print('====================query data of {}========================'.format(vehicle_name))
            original_data_dir = 'original_data'
            orgnized_throttle_data_dir = 'orgnized_throttle_data'
            orgnized_traction_data_dir = 'orgnized_traction_data'
            original_data_list = []
            orgnized_throttle_data_list = []
            orgnized_traction_data_list = []
            
            time_pair_list = []
            start_t = datetime.strptime(time_interval[0], '%Y-%m-%d %H:%M:%S')
            end_t = datetime.strptime(time_interval[1], '%Y-%m-%d %H:%M:%S')
            print('========{}~{}========'.format(time_interval[0], time_interval[1]))
            temp_t = start_t
            while(temp_t + timedelta(days = 1) < end_t):
                time_pair_list.append([temp_t, temp_t + timedelta(hours = 12)])
                temp_t += timedelta(days = 1)
            time_pair_list.append([temp_t, end_t])
            print('time_pair_list:\n{}'.format(time_pair_list))
            
            for time_pair in time_pair_list:
                # self.search_ground_truth(vehicle_name, time_interval)
                start_ts = time_pair[0] - self._time_gap
                end_ts = time_pair[1] - self._time_gap
                weight_gt = self.get_weight_gt(weight_cursor, vehicle_name, start_ts)
                
                print('========{}~{}, weight = {}, vehicle:{}========'.format(time_pair[0], time_pair[1], weight_gt, vehicle_name))
                # if(weight_gt == 0):
                #     # print('weight_gt is null')
                #     continue
                total_minutes = (end_ts - start_ts).total_seconds() / 60
                temp_ts = start_ts
                save_start_ts = start_ts
                
                df = pd.DataFrame(columns=QUERY_DATA)
                while temp_ts + query_delta_time <= end_ts:
                    weather_info = self.get_weather_info(weather_cursor, vehicle_name, temp_ts, temp_ts + query_delta_time)
                    if(len(weather_info) == 0):
                        temp_ts += query_delta_time
                        self._weather_invalid_num += 1
                        # print('weather info is null')
                        continue
                    self.data_process(vehicle_name, weight_gt, weather_info, temp_ts, temp_ts + query_delta_time,
                                      original_data_list,
                                      orgnized_throttle_data_list,
                                      orgnized_traction_data_list)
                    temp_ts += query_delta_time
                    
                    sub_progress = (temp_ts - start_ts).total_seconds() / 60 / total_minutes * 100
                    total_progress = (temp_ts - start_ts).total_seconds() / 60 / self._total_minutes * 100
                    cumsumed_time = datetime.now() - start_time
                    remaining_time = cumsumed_time * int(100/total_progress)
                    print("SubProgress: {:.2f}%, TotalProgress: {:.2f}%, cumsumed_time: {}, remaining_time: {}".format(sub_progress, total_progress, cumsumed_time, remaining_time), end="\r")
                    sys.stdout.flush()

                    if((temp_ts - save_start_ts) >= save_delta_time):
                        save_start_ts = temp_ts
                        # self.save_data(original_data_list,original_data_dir)
                        self.save_data(orgnized_throttle_data_list,orgnized_throttle_data_dir)
                        orgnized_throttle_data_list = []
                        self.save_data(orgnized_traction_data_list,orgnized_traction_data_dir)
                        orgnized_traction_data_list = []
                        
                    
                if (temp_ts + query_delta_time > end_ts):
                    self.data_process(vehicle_name, weight_gt, weather_info, temp_ts, end_ts, 
                                      original_data_list,
                                      orgnized_throttle_data_list,
                                      orgnized_traction_data_list)
                    # self.save_data(original_data_list,original_data_dir)
                    self.save_data(orgnized_throttle_data_list,orgnized_throttle_data_dir)
                    orgnized_throttle_data_list = []
                    self.save_data(orgnized_traction_data_list,orgnized_traction_data_dir)
                    orgnized_traction_data_list = []

        
        print('_throttle_train_num:{}, _throttle_test_num:{}, _throttle_validation_num:{}'.format(self._throttle_train_num,
                                                                                                self._throttle_test_num,
                                                                                                self._throttle_validation_num)) 
             
        print('_traction_train_num:{}, _traction_test_num:{}, _traction_validation_num:{}'.format(self._traction_train_num,
                                                                                                self._traction_test_num,
                                                                                                self._traction_validation_num)) 
        
        throttle_valid_num = self._throttle_train_num + self._throttle_test_num + self._throttle_validation_num
        traction_valid_num = self._traction_train_num + self._traction_test_num + self._traction_validation_num
        print('total num:{}, throttle valid num:{}, traction valid num:{}, weather invalid num:{}'.format(self._total_num,
                                                                                                        throttle_valid_num,
                                                                                                        traction_valid_num,
                                                                                                        self._weather_invalid_num)) 
                          
def main(args):
    data_collector = DataCollector(args)
    data_collector.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data Collector')
    dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
    print('file path:{}'.format(dir_path))
    parser.add_argument('--config', default=os.path.join(dir_path,'config/data_query_for_dynamic_model_config.yaml'), type=str)
    # parser.add_argument('--output-dir', default=os.path.join(dir_path,'data/backup_data/dynamic_model'))
    parser.add_argument('--output-dir', default='/mnt/intel/jupyterhub/jianhao.dong/dynamic_model1')
    args = parser.parse_args()
  
    main(args)
