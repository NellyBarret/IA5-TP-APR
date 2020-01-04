import math
from collections import namedtuple
from random import random

import gym
import matplotlib.pyplot as plt
import torch.nn as nn


class ReplayAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.mem = ReplayMemory(10000)

    def act(self, observation, reward, done):
        return self.action_space.sample()


class Interaction:
    def __init__(self, state, action, next_state, reward, done):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.done = done


# memoire = [(state, action, reward, next_state, done), ...]
class ReplayMemory:
    def __init__(self, capacite):
        self.capacite = capacite
        self.memory = [None for _ in range(self.capacite)]  # initialisation de la mémoire avec des None
        self.position = 0
        self.taille_occupee = 0

    def add_to_memory(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.capacite:
            self.memory.pop(0)  # on enleve les plus vieilles interactions
        self.memory[self.position] = Interaction(state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacite
        self.taille_occupee += 1

    def sample(self, taille):
        # tirage d'un échantillon parmi la mémoire
        return random.sample(self.memory, taille)

    def __len__(self):
        return len(self.memory)


class DQLearning(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(5, 4)  # transformation lineaire, cree automatiquement les poids et les biais
        self.output = nn.Linear()
        self.sigmoid = nn.Sigmoid()  # fonction d'activation
        self.relu = nn.ReLU()  # activation
        self.softmax = nn.Softmax(dim=1)
        self.memory = ReplayMemory(10000)
        self.liste_actions = []
        self.meilleure_action = 0
        self.strategie_choisie = "greedy"

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

    def exploration_greedy(self, epsilon):
        rand = random.random()
        if rand < epsilon:
            return self.liste_actions[random*self.liste_actions]
        else:
            return self.liste_actions[self.meilleure_action]

    # tau = stochasticité
    def exploration_boltzmann(self, tau, action):
        somme = 0
        for i in range(max_i):
            expo = math.exp(Q[state][action]/tau)  # calcul de l'exponentielle
            somme += expo
        proba = (math.exp(action/tau)) / ()
        return self.liste_actions[]

    def appliquer_strategie(self, strategie):
        if strategie == "greedy" or strategie == "boltzmann":
            self.strategie_choisie = strategie

        else:
            print("Mauvaise strategie")
            return


env = gym.make("CartPole-v1")
observation = env.reset()


# 2.1 - question 1
def cartpole_random(episode_count=100):
    list_reward = []
    list_iteration = []

    agent = ReplayAgent(env.action_space)
    reward = 0
    done = False

    for episode in range(episode_count):
        state = env.reset()
        somme_reward = 0
        while True:  # while not done s'arrete tout de suite
            action = agent.act(state, reward, done)
            next_state, reward, done, _ = env.step(action)
            somme_reward += reward
            if done:
                print("episode ", episode, ", somme reward : ", somme_reward)
                list_reward.append(somme_reward)
                list_iteration.append(episode)
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


# def cartpole_dqn(episode_count=100, gamma):
#     agent = ReplayAgent(env.action_space)
#     dql = DQLearning()
#     for episode in range(episode_count):
#         while True:
            # if done == True:
            #     jtheta = (Q[state][action] - reward + gamma)**2
            # else:
            #     jtheta = 0 # TODO
# def cartpole_dqn(episode_count=100):
#     list_reward = []
#     list_iteration = []
#
#     agent = ReplayAgent(env.action_space)
#     reward = 0
#     done = False
#     i = 0
#     for episode in range(episode_count):
#         state = env.reset()
#         somme_reward = 0
#         while True:
#             action = agent.act(state, reward, done)
#             next_state, reward, done, _ = env.step(action)
#             agent.mem.add_to_memory(state, action, reward, next_state, done)
#             state = next_state
#             i += 1
#             if done:
#                 print("episode: {}/{}, score: {}".format(episode, episode_count, i))
#                 break

# model_DQN = DQLearning()
# for i in range(episode_count):
#     ob = env.reset()
#     while not done:
#         action = agent.act(ob, reward, done) # action = 0 (gauche) ou 1 (droite) ; next_state : valeurs possibles d'états ; done : booleen pour dire si le jeu est fini ou non
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

list_iteration, list_reward = cartpole_random(episode_count=100)
cartpole_random_plot(list_iteration, list_reward)

# cartpole_dqn()

env.close()
# def cartpole_dqn():
