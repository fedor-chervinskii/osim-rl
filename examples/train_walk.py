# Derived from keras-rl
# import opensim as osim
import os

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import Callback, ModelIntervalCheckpoint
from keras.callbacks import TensorBoard

from osim.env import L2M2019Env

import argparse
import pickle
import numpy as np


PREFIX = 'snapshots/ddpg_simplified_vtgt_L2M2019Env'
LENGTH0 = 1.
obs_size = 99


def get_observation(obs_dict):
    # Augmented environment from the L2R challenge
    res = []

    # target velocity field (in body frame)
    #v_tgt = np.ndarray.flatten(obs_dict['v_tgt_field'])
    v_tgt = np.array(obs_dict['v_tgt_field'])
    res += v_tgt[:, 0, 0].tolist()

    res.append(obs_dict['pelvis']['height'])
    res.append(obs_dict['pelvis']['pitch'])
    res.append(obs_dict['pelvis']['roll'])
    res.append(obs_dict['pelvis']['vel'][0])
    res.append(obs_dict['pelvis']['vel'][1])
    res.append(obs_dict['pelvis']['vel'][2])
    res.append(obs_dict['pelvis']['vel'][3])
    res.append(obs_dict['pelvis']['vel'][4])
    res.append(obs_dict['pelvis']['vel'][5])

    for leg in ['r_leg', 'l_leg']:
        res += obs_dict[leg]['ground_reaction_forces']
        res.append(obs_dict[leg]['joint']['hip_abd'])
        res.append(obs_dict[leg]['joint']['hip'])
        res.append(obs_dict[leg]['joint']['knee'])
        res.append(obs_dict[leg]['joint']['ankle'])
        res.append(obs_dict[leg]['d_joint']['hip_abd'])
        res.append(obs_dict[leg]['d_joint']['hip'])
        res.append(obs_dict[leg]['d_joint']['knee'])
        res.append(obs_dict[leg]['d_joint']['ankle'])
        for MUS in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
            res.append(obs_dict[leg][MUS]['f'])
            res.append(obs_dict[leg][MUS]['l'])
            res.append(obs_dict[leg][MUS]['v'])
    return res


def save_memory(memory, filename):
    mem = (memory, memory.actions,
           memory.rewards,
           memory.terminals,
           memory.observations)
    pickle.dump(mem, open(filename, "wb"), protocol=-1)  # highest protocol means binary format


def load_memory(filename):
    (memory, memory.actions,
     memory.rewards,
     memory.terminals,
     memory.observations) = pickle.load(open(filename, "rb"))
    return memory


class MemoryIntervalCheckpoint(Callback):
    def __init__(self, filepath, interval, verbose=0):
        super(MemoryIntervalCheckpoint, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.verbose = verbose
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        """ Save weights at interval steps during training """
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            # Nothing to do.
            return

        filepath = self.filepath.format(step=self.total_steps, **logs)
        if self.verbose > 0:
            print('Step {}: saving model to {}'.format(self.total_steps, filepath))
        save_memory(agent.memory, self.filepath)


# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--steps', dest='steps', action='store', default=10000, type=int)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--memory', dest='memory', action='store', default=PREFIX + "_memory.pkl")
args = parser.parse_args()

# set to get observation in array

# Load walking environment
class MyEnv(L2M2019Env):
    env = L2M2019Env(visualize=False)
    def reset(self, **kwargs):
        obs_dict = self.env.reset()
        return get_observation(obs_dict)
    def step(self, action, **kwargs):
        obs_dict, reward, done, info = self.env.step(action)
        return get_observation(obs_dict), reward, done, info

env = MyEnv()
#env = Arm2DVecEnv(visualize=True)
env.reset()
#env.reset(verbose=True, logfile='arm_log.txt')

nb_actions = env.action_space.shape[0]

# Total number of steps in training
nallsteps = args.steps

# Create networks for DDPG
# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + (obs_size,)))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + (obs_size,), name='observation_input')
flattened_observation = Flatten()(observation_input)
x = concatenate([action_input, flattened_observation])
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Set up the agent for training
memory = SequentialMemory(limit=1000000, window_length=1)
#memory = load_memory(args.memory)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0.0, sigma=.5, size=env.osim_model.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.


def build_callbacks():
    checkpoint_weights_filename = PREFIX + '_weights_{step}.h5f'
    checkpoint_memory_filename = args.memory
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    callbacks += [MemoryIntervalCheckpoint(checkpoint_memory_filename, interval=1000)]
    callbacks += [TensorBoard(log_dir='./logs', histogram_freq=0,
                              write_graph=False, write_grads=False, write_images=False,
                              embeddings_freq=0, update_freq='epoch')]
    return callbacks

try:
    agent.load_weights(args.model)
    print("loaded weights from {}".format(args.model))
except:
    pass
agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=5000,
          log_interval=1000, callbacks=build_callbacks())
