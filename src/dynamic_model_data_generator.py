#!/usr/bin/env python
import argparse
import math
import os
import numpy as np
import pandas as pd
from collections import OrderedDict 
import yaml
from tools.utils.report_plotter import ReportPlotter
from utils.vehicle_dynamic_model import *

SAMPLE_TIME = 0.05

class VehicleDynamicModelGenerator:
    def __init__(self, config, init_v = 0.0):
        self._vehicle_dynamic_model = VehicleDynamicModel(config, init_v)
        
    def compute_output(self, u):
        self._vehicle_dynamic_model.update(u)
        return self._vehicle_dynamic_model.get_state()
    
    def reset(self, init_v = 0.0):
        self._vehicle_dynamic_model.reset(init_v)

    def generate_input(self, time_range, 
                       throttle_param,
                       pitch_param,
                       drag_coefficient_param, 
                       rolling_coefficient_param):
        u = []
        length = int(time_range / SAMPLE_TIME)
        for j in range(length):
            u_k = {}
            
            throttle = 0.6 * math.sin(2 * math.pi * SAMPLE_TIME * j  / throttle_param[0] - throttle_param[1]) + 0.2 * math.sin(2 * math.pi * SAMPLE_TIME * j / throttle_param[2]- throttle_param[3]) + throttle_param[4]
            throttle = max(min(throttle, 1), 0)
            gear_ratio = 1.0
            friction_torque = 0.07
            vehicle_mass = 37500
            pitch = -(0.4 * math.sin(2 * math.pi * SAMPLE_TIME * j / pitch_param[0] - pitch_param[1]) + 0.1 * math.sin(2 * math.pi * SAMPLE_TIME * j / pitch_param[2] -  - pitch_param[3])) / 20
            
            drag_coefficient =  (0.8 * math.sin(2 * math.pi * SAMPLE_TIME * j / drag_coefficient_param[0] - drag_coefficient_param[1]) + 
                                 0.2 * math.sin(2 * math.pi * SAMPLE_TIME * j / drag_coefficient_param[2] - drag_coefficient_param[3])) / 2.5 + drag_coefficient_param[4]
            # drag_coefficient = 0.0
            rolling_friction_coefficient =  (0.8 * math.sin(2 * math.pi * SAMPLE_TIME * j / rolling_coefficient_param[0] - rolling_coefficient_param[1] ) + 
                                             0.2 * math.sin(2 * math.pi * SAMPLE_TIME * j / rolling_coefficient_param[2] - rolling_coefficient_param[3] )) / 200 + rolling_coefficient_param[4]
            # rolling_friction_coefficient = 0.012
            u_k[THROTTLE] = throttle
            u_k[GEAR_RATIO] = gear_ratio
            u_k[FRICTION] = friction_torque
            u_k[MASS] = vehicle_mass
            u_k[PITCH] = pitch
            u_k[DRAG_COEFFICIENT] = drag_coefficient
            u_k[ROLLING_COEFFICIENT] = rolling_friction_coefficient
            u.append(u_k)
        return u


def plot_signal(report_plotter, output_dir, output_name, data_dict):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for segment_name, signals in data_dict.items():
        print(' plot {}, type {}, size {}'.format(segment_name, type(signals), signals.shape))
        subplot_figure = None
        plot_html_str = ""
        figure_list = []
        for signal_name, data in signals.iteritems():
            print(' plot {}, size: {}'.format(signal_name, len(data)))
            subplot = report_plotter.plot_figure_plotly(x_list = [np.array(range(len(data)))], 
                                                y_list = [np.array(data)],
                                                legend_list = ['{}'.format(signal_name)],
                                                x_label = 'timestamp',
                                                y_label = '{}'.format(signal_name),
                                                title = '{}'.format(signal_name),
                                                legend_prefix = '',
                                                figure_height=300,)
            figure_list.append(subplot)

        subplot_figure_list = [(i + 1, 1, fig) for i, fig in enumerate(figure_list)]
        subplot_figure = report_plotter.append_figure_to_subplot_plotly(subplot_figure_list, 
                                                                            row_num = len(figure_list), 
                                                                            col_num = 1, 
                                                                            template="plotly_dark", 
                                                                            subplot_fig=subplot_figure,
                                                                            figure_height = 300,
                                                                            vertical_spacing = 0.02)
        plot_html_str += report_plotter.get_fuel_fig_html_str({output_name: subplot_figure})
        html_str = report_plotter.generate_html_fuel_report(plot_html_str)
        with open(os.path.join(output_dir,'{}_{}.html'.format(output_name, segment_name)), 'w') as f:
            f.write(html_str)


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    time_range = config['time_range']

    vehicle_dynamic_data_generator = VehicleDynamicModelGenerator(config['vehicle_param'], config['init_v'])
    report_plotter = ReportPlotter('ReportGenerator')

    # phase, frequency, phase, frequency, offset
    throttle_params = [
        [25, 0, 40, 0, 0.2],
        [25, 0, 40, 0, 0.2],
        [25, 0, 40, 0, 0.2],
        [25, 0, 40, 0, 0.2],  
    ]

    # phase, frequency, phase, frequency
    pitch_params = [
        [100,   math.pi/4,      200,    math.pi/4],
        [100,   math.pi/2,      200,    math.pi/2],
        [100,   math.pi*3/4,    200,    math.pi*3/4],
        [100,   0,              200,    0],  
    ]

    # # phase, frequency, phase, frequency, offset
    # throttle_params = [
    #     [25, 0, 40, 0, 0.2], 
    # ]

    # # phase, frequency, phase, frequency
    # pitch_params = [
    #     [100,   math.pi/4,      200,    math.pi/4],
    # ]
    
    # phase, frequency, phase, frequency, offset
    drag_coefficient_params = [
        [200,   0,          400,    math.pi/4,  0.8],
        [300,   math.pi/4,  600,    math.pi/2,  0.6],
        [100,   math.pi/2,  300,    math.pi/4,  0.4],
        [200,   math.pi/2,  100,    0,          1.0],  
    ]


    # phase, frequency, phase, frequency, offset
    rolling_coefficient_params = [
        [300,   math.pi/2,    500,    0,          0.006],
        [400,   0,            800,    math.pi/4,  0.006],
        [200,   math.pi*3/4,  400,    math.pi/4,  0.006],
        [300,   0,            300,    math.pi/2,  0.005],
    ]
    df_dict = OrderedDict()
    input_dict = OrderedDict()
    for m in range(len(throttle_params)):
        for n in range(len(drag_coefficient_params)):
            output_index = m * len(drag_coefficient_params) + n
            print('========{} data============'.format(output_index))
            vehicle_dynamic_data_generator.reset(config['init_v'])
            u = vehicle_dynamic_data_generator.generate_input(time_range,
                                                            throttle_params[m],
                                                            pitch_params[m],
                                                            drag_coefficient_params[n],
                                                            rolling_coefficient_params[n])
            df = pd.DataFrame()
            raw_input = pd.DataFrame()
            for i in range(len(u)):   
                new_df = vehicle_dynamic_data_generator.compute_output(u[i])
                # print(new_df)
                df = pd.concat([df, new_df], ignore_index=True)
                # print('{}:{}'.format(j, type(u[j])))
                new_input = pd.DataFrame(u[i], index = [0])
                # print('{}:{}'.format(j, new_input.shape))
                raw_input = pd.concat([raw_input, new_input], ignore_index=True)

            df_dict['{}'.format(output_index)] = df
            output_dir = os.path.join(args.output_dir,'output','data')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            df.to_csv(os.path.join(output_dir,'data_{:02d}.csv'.format(output_index)), index=None)

            
            input_dict['{}'.format(output_index)] = raw_input
            raw_input_dir = os.path.join(args.output_dir,'raw_input', 'data')
            if not os.path.exists(raw_input_dir):
                os.makedirs(raw_input_dir)
            raw_input.to_csv(os.path.join(raw_input_dir,'raw_input_{}.csv'.format(output_index)), index=None)        
    plot_signal(report_plotter, os.path.join(args.output_dir,'output', 'signal_plot'), 'signal',df_dict)
    plot_signal(report_plotter, os.path.join(args.output_dir,'raw_input','signal_plot'), 'signal',input_dict)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sim Vehicle Dynamic Model')
    dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
    print('file path:{}'.format(dir_path))
    parser.add_argument('--config', default=os.path.join(dir_path,'config/dynamic_model_data_generator_config.yaml'), type=str)
    parser.add_argument('--output-dir', default=os.path.join(dir_path,'data/backup_data/sim_dynamic_model'))
    args = parser.parse_args()
  
    main(args)