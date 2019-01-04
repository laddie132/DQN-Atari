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

# Plotting for Q1, performance on Pong for lr=1 on images for 4.2m steps
pixel_data = pickle.load(open('data/pong_double_6000k_data.pkl','rb'))
pixel_t = pixel_data['t_log']
pixel_mean_rewards = pixel_data['mean_reward_log']
pixel_best_rewards = pixel_data['best_mean_log']

pixel_plot= plt.figure()
pixel_mean_rew, = plt.plot(pixel_t, pixel_mean_rewards, label='Mean 100-Episode Reward')
pixel_best_rew, = plt.plot(pixel_t, pixel_best_rewards, label='Best Mean Reward')
plt.title('Q-Learning Performance on Pong with Pixels')
plt.xlabel('Timesteps')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('Reward')
plt.legend(loc=4)
pp = PdfPages('pong_ram_plot.pdf')
pp.savefig(pixel_plot)
pp.close()