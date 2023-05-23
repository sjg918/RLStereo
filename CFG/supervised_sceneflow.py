
from easydict import EasyDict
import torch

cfg = EasyDict()

cfg.devices = [0, 1]
cfg.num_cpu = 4

# directory
cfg.sceneflow_home = '/home/gpuadmin/Dataset/sceneflow/'
cfg.logdir = '/home/gpuadmin/kew/RL-SM/checkpoint/supervised_sceneflow/'
cfg.traintxt = '/home/gpuadmin/kew/RL-SM/gendata/sceneflow_train.txt'
cfg.valtxt = '/home/gpuadmin/kew/RL-SM/gendata/sceneflow_test.txt'

# training
cfg.initial_disp = 96
cfg.max_action = 32
cfg.max_iteration = 6
cfg.max_disp = 192

cfg.batchsize = 8
cfg.learning_rate = 0.0001
cfg.epoch_milestones_0 = 0
cfg.maxepoch = 20
cfg.epoch_milestones_1 = 10
cfg.epoch_milestones_2 = 21
def burnin_schedule_(i):
    if i < cfg.epoch_milestones_0:
        factor = pow(i / cfg.stereo_milestones_0, 4)
    elif i < cfg.epoch_milestones_1:
        factor = 1
    elif i < cfg.epoch_milestones_2:
        factor = 0.1
    else:
        factor = 0.25
    return factor
cfg.burnin_schedule = burnin_schedule_
