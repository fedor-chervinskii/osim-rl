import sys
sys.path = [''] + sys.path
from osim.env import L2M2019Env

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import argparse

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--episodes', type=int, default=5)
args = parser.parse_args()

env = L2M2019Env(visualize=True)

nb_actions = env.action_space.shape[0]

# Total number of steps in training

# Create networks for DDPG
# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
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
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
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
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0.5, sigma=.5, size=env.osim_model.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=2000, nb_steps_warmup_actor=2000,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)

agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
agent.load_weights(args.model)

scores = []

for i in range(args.episodes):
    total_reward = 0
    t = 0
    i = 0
    observation = env.reset()
    while True:
        i += 1
        action = agent.forward(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    print('score={} steps={}'.format(total_reward, i))
    scores.append(total_reward)
print("average score: {}".format(np.mean(scores)))