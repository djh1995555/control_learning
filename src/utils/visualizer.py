#!/usr/bin/env python
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.report_plotter import ReportPlotter

class Visualizer():
    def __init__(self, config):
        self._config = config
        self._report_plotter = ReportPlotter('ReportGenerator')
            
    def plot_loss(self, output_dir, loss):
        width = self._config['loss_fig_width']
        height = self._config['loss_fig_height']
        zoom = self._config['loss_fig_zoom']
        width *= zoom
        height *= zoom

        plt.figure(1, figsize = (width,height))
        plt.plot(loss, 'b', label='loss')
        plt.title('train loss')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_dir,'train_loss.png'))
        
    def plot_comparison_result(self, comparison_result, output_dir, output_name):
        plots_in_one_file = 30
        result_num = math.floor(len(comparison_result) / plots_in_one_file) + 1
        for j in range(result_num):
            subplot_figure = None
            plot_html_str = ""
            figure_list = []
            for i in range(plots_in_one_file):
                index = i + j * plots_in_one_file
                # print('index:{}'.format(index))
                if(index >= len(comparison_result)):
                    break
                
                legend_list = []
                value_list = []
                epoch_list = []
                name = comparison_result[index][0]
                target = np.array(comparison_result[index][1][1])
                output = np.array(comparison_result[index][1][0])
                
                value_list.append(target)
                epoch_list.append(np.array(range(len(target))))                
                value_list.append(output)
                epoch_list.append(np.array(range(len(output))))
                
                legend_list.append('target of {} {}'.format(name, index))
                legend_list.append('output of {} {}'.format(name, index))
                is_train_result = 'train' in output_name
                if(self._config['plot_benchmark'] and not is_train_result):
                    benchmark = np.array(comparison_result[index][1][2])
                    value_list.append(benchmark)
                    epoch_list.append(np.array(range(len(benchmark))))
                    legend_list.append('benchmark of {} {}'.format(name, index))
                subplot = self._report_plotter.plot_figure_plotly(x_list = epoch_list, 
                                                    y_list = value_list,
                                                    legend_list = legend_list,
                                                    x_label = 'timestamp',
                                                    y_label = 'output',
                                                    title = 'comparison of {} {}'.format(name, index),
                                                    legend_prefix = '',
                                                    figure_height=self._config['comparison_fig_height'],)
                figure_list.append(subplot)

            subplot_figure_list = [(i + 1, 1, fig) for i, fig in enumerate(figure_list)]
            subplot_figure = self._report_plotter.append_figure_to_subplot_plotly(subplot_figure_list, 
                                                                                row_num = len(figure_list), 
                                                                                col_num = 1, 
                                                                                template="plotly_dark", 
                                                                                subplot_fig=subplot_figure,
                                                                                figure_height = self._config['comparison_fig_height'],
                                                                                vertical_spacing = self._config['vertical_spacig'])
            plot_html_str += self._report_plotter.get_fuel_fig_html_str({output_name: subplot_figure})
            html_str = self._report_plotter.generate_html_fuel_report(plot_html_str)
            with open(os.path.join(output_dir, '{}_{}.html'.format(output_name, j)), 'w') as f:
                f.write(html_str)