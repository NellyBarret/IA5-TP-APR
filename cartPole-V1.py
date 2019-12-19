from collections import namedtuple
from random import random

import gym
import matplotlib.pyplot as plt
import torch.nn as nn
#Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        # self.mem = ReplayMemory(10000)

    def act(self, observation, reward, done):
        return self.action_space.sample()


# memoire = [(state, action, reward, next_state, don), ...]
# class ReplayMemory:
#     def __init__(self, capacite):
#         self.capacite = capacite
#         self.memory = [0 for _ in range(self.capacite)] # initialisation de la mémoire à 0
#         self.position = 0
#         self.taille_occupee = 0
#
#     def add_to_memory(self, *args):
#         if len(self.memory) < self.capacite:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) * self.capacite
#         self.taille_occupee += 1
#
#     def sample(self, taille):
#         # tirage d'un échantillon parmi la mémoire
#         return random.sample(self.memory, taille)
#
#     def __len__(self):
#         return len(self.memory)
#
#
# class DQLearning(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear()  # transformation lineaire, cree automatiquement les poids et les biais
#         self.output = nn.Linear()
#         self.sigmoid = nn.Sigmoid()  # fonction d'activation
#         self.relu = nn.ReLU()  # activation
#         self.softmax = nn.Softmax(dim=1)
#         self.memory = ReplayMemory(10000)
#
#     def forward(self, x):
#         x = self.hidden(x)
#         x = self.sigmoid(x)
#         x = self.output(x)
#         x = self.softmax(x)
#         return x


env = gym.make("CartPole-v1")
observation = env.reset()

# list_iteration = []
# agent = ReplayAgent(env.action_space)
#model_DQN = DQLearning()

# for i in range(episode_count):
#     ob = env.reset()
#     while not done:
#         action = agent.act(ob, reward, done)
#         # action = 0 (gauche) ou 1 (droite) ; next_state : valeurs possibles d'états ; done : booleen pour dire si le jeu est fini ou non
#         next_state, reward, done, _ = env.step(action)
#         agent.mem.add_to_memory(ob, action, next_state, reward)  # on ajoute la transition a la memoire de l'agent
#         somme_reward += reward
#         # state = next_state  # on va sur l'état suivant
#         # if model_DQN.memory.taille_occupee > model_DQN.memory.capacite:
#             # model_DQN.learn()
#         if done:
#             print("episode ", i, ", somme reward : ", somme_reward)
#             list_reward.append(somme_reward)
#             list_iteration.append(i)
#
#         break
# for i in range(1000):
#     env.render()
#     action = agent.act(ob, reward, done) # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)
#     somme_reward += reward
#
#     if done:
#         print("episode ", i, ", somme reward : ", somme_reward)
#         list_reward.append(somme_reward)
#         list_iteration.append(i)
#         somme_reward = 0
#         observation = env.reset()
# env.close()


def cartpole_random(episode_count=100):
    observation = env.reset()
    list_reward = []
    list_iteration = []
    somme_reward = 0

    agent = ReplayAgent(env.action_space)
    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        somme_reward = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            somme_reward += reward
            if done:
                print("episode ", i, ", somme reward : ", somme_reward)
                list_reward.append(somme_reward)
                list_iteration.append(i)
                break
    return list_iteration, list_reward


def cartpole_random_plot(list_iteration, list_reward):
    fig = plt.figure()
    fig.subplots_adjust(top=0.8)
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('somme des récompenses')
    ax1.set_xlabel('épisode')
    ax1.set_title('cartPole - v1')
    ax1.plot(list_iteration, list_reward)
    plt.show()


list_iteration, list_reward = cartpole_random(episode_count=100)
print(list_iteration, list_reward)
cartpole_random_plot(list_iteration, list_reward)
env.close()


# def cartpole_dqn():
