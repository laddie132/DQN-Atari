#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

sns.set_style('whitegrid')

def plot_q1():
    pixel_data = pickle.load(open('data/pong_double_7200k_data.pkl', 'rb'))
    pixel_t = pixel_data['t_log']
    pixel_mean_rewards = pixel_data['mean_reward_log']
    pixel_best_rewards = pixel_data['best_mean_log']

    pixel_plot = plt.figure()
    pixel_mean_rew, = plt.plot(pixel_t, pixel_mean_rewards, label='Mean 100-Episode Reward')
    pixel_best_rew, = plt.plot(pixel_t, pixel_best_rewards, label='Best Mean Reward')
    plt.title('Q-Learning Performance on Pong with Pixels')
    plt.xlabel('Timesteps')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ylabel('Reward')
    plt.legend(loc=4)
    pp = PdfPages('pong_double_plot.pdf')
    pp.savefig(pixel_plot)
    pp.close()


def plot_q2(files, labels):
    assert len(files) == len(labels)
    all_plot = plt.figure()

    for i, f in enumerate(files):
        pixel_data = pickle.load(open(f, 'rb'))
        pixel_t = pixel_data['t_log']
        pixel_mean_rewards = pixel_data['mean_reward_log']
        pixel_best_rewards = pixel_data['best_mean_log']

        plt.plot(pixel_t, pixel_mean_rewards, label=labels[i])

    plt.title('Learning Rate vs. Q-Learning Performance for Pong with Pixel')
    plt.xlabel('Timesteps')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ylabel('Reward')
    plt.legend(loc=2)
    pp = PdfPages('lr_plot.pdf')
    pp.savefig(all_plot)
    pp.close()


if __name__ == '__main__':
    plot_q1()
    # plot_q2(files=['ram_lr01/ram_lr0.1_1000000_data.pkl',
    #                'ram_lr1/ram_lr1_1000000_data.pkl'],
    #         labels=['LR = 100',
    #                 'LR = 10'])