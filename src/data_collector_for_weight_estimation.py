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
import matplotlib.pyplot as plt
from tqdm import tqdm
from subprocess import PIPE, STDOUT
from collections import OrderedDict
from utils.topics_and_signals import *
from datetime import datetime, timedelta
from utils.report_plotter import ReportPlotter
from utils.vehicle_dynamic_model import VehicleDynamicModel
from clickhouse_driver import Client

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
                WEIGHT,
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
                WEIGHT,
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
    WHERE vehicle=%(vehicle)s
    and ts >= %(start_ts)s
    and ts <= %(end_ts)s
    ORDER by ts """
)


class DataCollector:
    def __init__(self,args):
        with open(args.config, 'r') as f:
            self._config = yaml.load(f,Loader=yaml.FullLoader)
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
        clickhouse_config = DB_CONF["clickhouse"]
        self.client = Client(host=clickhouse_config["host"], port=clickhouse_config["port"], user=clickhouse_config["user"], password=clickhouse_config["password"], database=clickhouse_config["database"])

                   
    def data_query_from_database(self, vehicle_name, start_ts, end_ts):  
        temp_ts = start_ts
        query_delta_time = timedelta(minutes=self._config['query_step'])
        df = pd.DataFrame(columns=QUERY_DATA)
        empty_frame_num = 0
        while temp_ts + query_delta_time <= end_ts:
            try:
                df_new = self.client.execute(SQL_BAG_MSG,{"vehicle": vehicle_name,"start_ts": temp_ts, "end_ts": temp_ts + query_delta_time})
            except Exception as e:
                temp_ts += query_delta_time
                continue
            
            if(len(df_new) != 0):
                empty_frame_num = 0
                new_df = pd.DataFrame(df_new,columns=QUERY_DATA)
                df = pd.concat([df, new_df], ignore_index=True, sort=SORT)
                # print("query {} with interval {}~{}/{}: data length = {}".format(vehicle_name, temp_ts + self._time_gap, 
                #                                                     temp_ts + query_delta_time + self._time_gap, 
                #                                                     end_ts + self._time_gap, len(df)))
            else:
                empty_frame_num += 1
                # print('no data queried!')
                if(empty_frame_num > 10):
                    return df
                    
            temp_ts += query_delta_time
            
        if (temp_ts + query_delta_time > end_ts):
            try:
                df_new = self.client.execute(SQL_BAG_MSG,{"vehicle": vehicle_name,"start_ts": temp_ts, "end_ts": end_ts})            
            except Exception as e:
                return df
            
            if(len(df_new) != 0):
                df = pd.concat([df, pd.DataFrame(df_new, columns=QUERY_DATA)], ignore_index=True, sort=SORT)
                # print("query interval {}~{}/{}: data length = {}".format(temp_ts + self._time_gap, 
                #                                                         end_ts + self._time_gap, 
                #                                                         end_ts + self._time_gap, len(df)))


        # print("data query finish")
        return df

    def plot_signal(self, output_dir, output_name, data_dict):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for segment_name, signals in data_dict.items():
            subplot_figure = None
            plot_html_str = ""
            figure_list = []
            for signal_name, data in signals.items():
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
        
    def orgnize_throttle_data(self, data):
        df = pd.DataFrame(columns=TARGET_DATA_1)
        df[WEIGHT] = data[WEIGHT]
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
    
    def orgnize_traction_data(self, data):
        df = pd.DataFrame(columns=TARGET_DATA_2)
        if((len(data[CLUTCH_SLIP]) > 0 and max(data[CLUTCH_SLIP]) > 5)):
            return df
        df[WEIGHT] = data[WEIGHT]
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
        if(len(data_list)==0):
            return
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
        # print("data_num: {}, train_data_range: {}, test_data_range:{}".format(data_num, train_data_range,test_data_range))
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
            for signal_name, signal_data in data.items():
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
        
        if(len(original_data) < self._config['query_delta_time'] * 600 ):
            return False
    
        try:
            original_data = original_data.drop(TS, axis=1)
            original_data = original_data.interpolate(method='linear')
        except TypeError:
            return False
        original_data = original_data.dropna()
        self._total_num += 1
        if(self.is_invalid_data(original_data)):
            return False
        
        label = {'vehicle': vehicle_name,
                 'start_ts': self.deal_with_time_string('{}'.format(start_ts + self._time_gap)),
                 'end_ts': self.deal_with_time_string('{}'.format(end_ts + self._time_gap)),
                 'mass': weight_gt,
                 'tire': 3,
                 'weather': weather_info[0][2]
                 }

        original_data = original_data.iloc[::self._config['de_sample']]
        original_data[WEIGHT] = weight_gt

        original_data_list.append((segment_name, label, original_data))
        
        orgnized_data = self.orgnize_throttle_data(original_data)
        orgnized_throttle_data_list.append((segment_name, label, orgnized_data))
        
        orgnized_data_2 = self.orgnize_traction_data(orgnized_data)
        if(len(orgnized_data_2) > 0):
            orgnized_traction_data_list.append((segment_name, label, orgnized_data_2))
        return True
    
    def get_weight_gt(self, vehcile_name, start_time):
        weight_gt_record_filepath = os.path.join(weight_gt_record_dir, '{}.csv'.format(vehcile_name))
        weight_gt_record = pd.read_csv(weight_gt_record_filepath)
        weight_gt_record['start_time'] = pd.to_datetime(weight_gt_record['start_time'])
        weight_gt_record['end_time'] = pd.to_datetime(weight_gt_record['end_time'])

        result_row = weight_gt_record[(weight_gt_record['start_time'] <= start_time) & (weight_gt_record['end_time'] >= start_time)]
        if(result_row.shape[0]==0):
            return np.nan
        return result_row['weight_gt'].iloc[0]
    
    def get_weather_info(self, cursor, vehicle_name, start_ts, end_ts):
        start_timestamp = float(time.mktime(start_ts.timetuple()))
        end_timestamp = float(time.mktime(end_ts.timetuple()))
        cursor.execute(WEATHER_QUERY_SQL.format(vehicle_name, start_timestamp, end_timestamp))
        records = cursor.fetchall()
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

        weather_conn_string = "host='labeling2' dbname='tractruck_new' user='tractruck_viewer' password='tractruck_viewer'"
        weather_conn = psycopg2.connect(weather_conn_string)
        weather_cursor = weather_conn.cursor()
        
        original_data_dir = 'original_data'
        orgnized_throttle_data_dir = 'orgnized_throttle_data'
        orgnized_traction_data_dir = 'orgnized_traction_data'
        original_data_list = []
        orgnized_throttle_data_list = []
        orgnized_traction_data_list = []
        
        start_ts = datetime.strptime(time_intervals[0][0], '%Y-%m-%d %H:%M:%S') - self._time_gap
        end_ts = datetime.strptime(time_intervals[0][1], '%Y-%m-%d %H:%M:%S') - self._time_gap
        temp_ts = start_ts
        save_start_ts = start_ts
        
        file_num = 0
        last_weight_gt = 0
        ts_pairs_list = []
        while temp_ts + query_delta_time <= end_ts:
            ts_pairs_list.append((temp_ts, temp_ts + query_delta_time))
            temp_ts += query_delta_time
        ts_pairs_list.append((temp_ts, end_ts))
        
        
        print(vehicle_names)
        vehicles_loop = tqdm(enumerate(vehicle_names), total=len(vehicle_names), desc="VEHICLE", leave=False)
        for idx, vehicle_name in vehicles_loop:
            iter_loop = tqdm(enumerate(ts_pairs_list), total=len(ts_pairs_list), desc="QUERY", leave=False)
            for i, ts_pair in iter_loop:
                iter_loop.set_description(f'Vehicle={vehicle_name}|start time={ts_pair[0] + self._time_gap}|end time={ts_pair[1] + self._time_gap}')
                # print("query {} with interval {}~{}".format(vehicle_name, ts_pair[0] + self._time_gap, ts_pair[1] + self._time_gap))
                weight_gt = self.get_weight_gt(vehicle_name, ts_pair[0] + self._time_gap)
                if(weight_gt != last_weight_gt):
                    file_num = 0
                    self.save_data(orgnized_throttle_data_list,orgnized_throttle_data_dir)
                    orgnized_throttle_data_list = []
                    self.save_data(orgnized_traction_data_list,orgnized_traction_data_dir)
                    orgnized_traction_data_list = []
                last_weight_gt = weight_gt
                
                if(file_num >= self._config['file_num_for_one_weight']):
                    continue
                
                if(np.isnan(weight_gt)):
                    continue 
                weather_info = self.get_weather_info(weather_cursor, vehicle_name, ts_pair[0] + self._time_gap, ts_pair[1] + self._time_gap)
                if(len(weather_info) == 0):
                    self._weather_invalid_num += 1
                    continue
                success = self.data_process(vehicle_name, weight_gt, weather_info, ts_pair[0], ts_pair[1],
                                    original_data_list,
                                    orgnized_throttle_data_list,
                                    orgnized_traction_data_list)
                file_num += 1 if(success) else 0
                # if(success):
                #     print('weight_gt:{}, weight_gt:{}, file_num:{}'.format(weight_gt, last_weight_gt, file_num))
                if((ts_pair[0] - save_start_ts) >= save_delta_time):
                    save_start_ts = ts_pair[0]
                    # self.save_data(original_data_list,original_data_dir)
                    self.save_data(orgnized_throttle_data_list,orgnized_throttle_data_dir)
                    orgnized_throttle_data_list = []
                    self.save_data(orgnized_traction_data_list,orgnized_traction_data_dir)
                    orgnized_traction_data_list = []
            # print('{} is done'.format(vehicle_name))
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
        self.client.disconnect()
                          
def main(args):
    data_collector = DataCollector(args)
    data_collector.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data Collector')
    dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
    weight_gt_record_dir = os.path.join(dir_path, 'weight_gt_record')
    parser.add_argument('--config', default=os.path.join(dir_path,'config/data_collector_for_weight_estimation_config.yaml'), type=str)
    parser.add_argument('--output-dir', default='/mnt/intel/jupyterhub/jianhao.dong/data/weight_estimation2')
    args = parser.parse_args()
  
    main(args)
