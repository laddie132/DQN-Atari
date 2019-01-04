#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

sns.set_style('whitegrid')


def plot_q1():
    pixel_data = pickle.load(open('data/pong_double_9500k_data.pkl', 'rb'))
    pixel_t = pixel_data['t_log']
    pixel_mean_rewards = pixel_data['mean_reward_log']
    pixel_best_rewards = pixel_data['best_mean_log']

    pixel_plot = plt.figure()
    plt.plot(pixel_t, pixel_mean_rewards, label='Mean 100-Episode Reward')
    plt.plot(pixel_t, pixel_best_rewards, label='Best Mean Reward')
    plt.title('Q-Learning Performance on Pong with Pixels')
    plt.xlabel('Timesteps')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ylabel('Reward')
    plt.legend(loc=4)
    pp = PdfPages('pong_double_plot.pdf')
    pp.savefig(pixel_plot)
    pp.close()


def plot_q2(files, labels, num_steps=None):
    assert len(files) == len(labels)
    all_plot = plt.figure()

    for i, f in enumerate(files):
        pixel_data = pickle.load(open(f, 'rb'))
        pixel_t = pixel_data['t_log'] if num_steps is None else pixel_data['t_log'][:num_steps]
        pixel_mean_rewards = pixel_data['mean_reward_log'] if num_steps is None \
            else pixel_data['mean_reward_log'][:num_steps]
        pixel_best_rewards = pixel_data['best_mean_log'] if num_steps is None \
            else pixel_data['best_mean_log'][:num_steps]

        plt.plot(pixel_t, pixel_mean_rewards, label=labels[i])

    plt.title('Learning Rate vs. Q-Learning Performance for Pong with Pixel')
    plt.xlabel('Timesteps')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ylabel('Reward')
    plt.legend(loc=2)
    pp = PdfPages('pong_hyper_parameters.pdf')
    pp.savefig(all_plot)
    pp.close()


if __name__ == '__main__':
    plot_q1()
    plot_q2(files=['data/pong_double_9500k_data.pkl',
                   'data/pong_3400k_data.pkl',
                   'data/pong_lr_1e-3_1600k_data.pkl'],
            labels=['Default Hyperparameters',
                    'No Double DQN',
                    'LR = 1e-3'],
            num_steps=None)
