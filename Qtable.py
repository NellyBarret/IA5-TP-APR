import math
from collections import namedtuple
import random
import numpy
import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


## MEMO
# action : 2 valeurs (droite / gauche)
# environment : [position of cart, velocity of cart, angle of pole, rotation rate of pole]
# state / next_state = tableau de 4 elements
# action : 0 ou 1
# reward : 1 si la baton n'est pas tombé
# done : True ou False
# policy: strategie pour choisir la meilleure action pour un état donné (celle qui permet de maximiser le gain)

# 2.3 - question 5
if __name__ == '__main__':
    # env = gym.make('CartPole-v1')  # creation de l'environnement
    env = gym.make('FrozenLake8x8-v0')
    nb_episodes = 5000  # 100
    # TODO: regarder pour env.observation_space.n
    q_table = numpy.zeros([env.observation_space.n, env.action_space.n])
    alpha = 0.1
    gamma = 0.9  # 0.6
    epsilon = 0.1
    eta = 0.628

    for i in range(nb_episodes):
        state = env.reset()
        while True:
            action = numpy.argmax(q_table[state, :] + numpy.random.randn(1, env.action_space.n)*(1./(i+1)))
            next_state, reward, done, _ = env.step(action)
            # mise a jour des q valeurs avec l'equation de bellman
            q_table[state, action] += eta * (reward + gamma*numpy.max(q_table[next_state, :]) - q_table[state, action])
            state = next_state
            if done:
                break
    print(q_table)
