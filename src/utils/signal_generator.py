#!/usr/bin/env python
import math
import os
import numpy as np
import pandas as pd
from tools.utils.report_plotter import ReportPlotter

SAMPLE_TIME = 0.05

class SignalGenerator:
    def __init__(self):
        self.reset()
        
    def compute_output(self, u_k, time_varying_variable):
        k = self._k
        y_k = self._y_k
        last_y_k = self._last_y_k
        last_u_k = self._last_u_k
        y = -0.85 * math.exp(-0.5 * k) * y_k + 0.25 * last_y_k + 0.5 * u_k + time_varying_variable * last_u_k * last_y_k
        self._k += 1
        self._last_y_k = self._y_k
        self._y_k = y
        self._last_u_k = u_k
        return y
    
    def reset(self):
        self._k = 0
        self._y_k = 0.0
        self._last_y_k = 0.0
        self._last_u_k = 0.0

    def generate_input(self, time_range):
        u = []
        length = int(time_range / SAMPLE_TIME)
        for j in range(length):
            u.append(0.8 * math.sin(math.pi * SAMPLE_TIME * j / 25) + 0.2 * math.sin(math.pi * SAMPLE_TIME * j / 40))   
        return u

def plot_signal(report_plotter, output_dir, output_name, data_dict):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for segment_name, signals in data_dict.items():
        subplot_figure = None
        plot_html_str = ""
        figure_list = []
        for signal_name, data in signals.items():
            # print(' plot {}, size: {}'.format(signal_name, len(data)))
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

def main():
    signal_generator = SignalGenerator()
    report_plotter = ReportPlotter('ReportGenerator')
    time_range = 500
    file_num = 10
    time_varying_variable_range = [0.5,0]
    u = signal_generator.generate_input(time_range)
    time_varying_variable_interval = float(abs(time_varying_variable_range[1] - time_varying_variable_range[0])) / file_num
    df_dict = {}
    for i in range(file_num):
        time_varying_variable = time_varying_variable_range[0] - i * time_varying_variable_interval
        print(time_varying_variable)
        signal_generator.reset()
        y = []
        for j in range(len(u)):
            y.append(signal_generator.compute_output(u[j], time_varying_variable))
        print('y_end:{}'.format(y[-1]))
        data = {'u': u, 'bias': [-1] * len(u), 'y': y}
        df = pd.DataFrame(data)
        df_dict['{}'.format(i)] = df
        dir_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')), 'data/simple_system')
        output_dir = os.path.join(dir_path,'data')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_csv(os.path.join(output_dir,'data_{}.csv'.format(i)), index=None)
    plot_signal(report_plotter, os.path.join(dir_path,'signal_plot'), 'signal',df_dict)
if __name__ == '__main__':
    main()