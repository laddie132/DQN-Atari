# 高级机器学习-作业5
在CS294-112 HW3基础上实现了Double DQN算法

## 目录
- data目录下保存的是不同参数下的训练奖励日志
- video目录下保存的是不同参数下的算法训练过程视频

## 使用
- 运行`run_dqn_atari.py`训练强化学习算法
- 运行`plot.py`绘制奖励曲线

> 以下是原仓库README
---
# CS294-112 HW 3: Q-Learning

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**
 * OpenCV
 * ffmpeg

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only files that you need to look at are `dqn.py` and `train_ac_f18.py`, which you will implement.

See the [HW3 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf) for further instructions.

The starter code was based on an implementation of Q-learning for Atari generously provided by Szymon Sidor from OpenAI.
