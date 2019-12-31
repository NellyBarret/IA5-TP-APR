import math
from collections import namedtuple
from random import random

import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


## MEMO
# action_sample : espace d'actions : 2 valuers (droite / gauche)
from tensorflow_core.python.keras.layers import Dense
from torch.optim import Adam


class DQNAgent:
    """
    Agent qui choisit des actions de manière aléatoire
    """
    def __init__(self, observation_space, action_space, epsilon, epsilon_maj, epsilon_min, gamma, batch_size):
        """
        Initialisation générale
        @param observation_space: TODO
        @param action_space: espace des actions possibles
        @param epsilon: taux d'exploration
        @param batch_size: TODO
        """
        self.action_space = action_space
        self.memory = []
        self.epsilon = epsilon
        self.epsilon_maj = epsilon_maj
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.model = nn.Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=self.epsilon))

    def remember(self, state, action, reward, next_state, done):
        """
        Ajoute une interaction à la mémoire de l'agent
        @param state: etat courant
        @param action: action effectuee
        @param reward: recompense recue de l'environnement
        @param next_state: etat dans lequel on arrive
        @param done: pour arreter l'agent quand il a fini
        """
        self.memory.append([state, action, reward, next_state, done])

    def act(self, state):
        """
        Choisit la meilleure action
        @param state: l'etat courant de l'agent
        """
        rand = random.random()
        if rand < self.epsilon:
            return random.randrange(self.action_space)
        q_valeurs = self.model.predict(state)
        return torch.argmax(q_valeurs[0])

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)  # creation du batch
        for state, action, reward, state_next, done in batch:
            q_valeurs_maj = reward
            if not done:
                q_valeurs_maj = (reward + self.gamma * torch.argmax(self.model.predict(state_next)[0]))
            q_valeurs = self.model.predict(state)
            q_valeurs[0][action] = q_valeurs_maj
            self.model.fit(state, q_valeurs)
        self.epsilon *= self.epsilon_maj
        self.epsilon = max(self.epsilon_min, self.epsilon)


# 2.1 - question 2
def evolution_rewards(liste_rewards):
    """
    Trace l'évolution de la somme des récompenses par épisodes
    """
    plt.plot([i for i in range(len(liste_rewards))], liste_rewards)
    plt.title("Evolution de la somme des récompenses par episodes")
    plt.show()


# 2.1 - question 1
if __name__ == '__main__':
    env = gym.make("CartPole-v1")  # creation de l'environnement
    agent = DQNAgent(env.action_space, env.observation_space, 0.001, 0.995, 0.01, 0.95, 20)  # creation de l'agent

    nb_episodes = 100

    all_rewards = []
    done = False  # pour savoir quand on s'arrete (le baton est tombé ou il est sorti de l'environnement)

    for i in range(nb_episodes):
        total_reward = 0
        env.render()
        state = env.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            agent.experience_replay()
    env.close()



#
# def cartpole_random_plot(list_iteration, list_reward):
#     fig = plt.figure()
#     fig.subplots_adjust(top=0.8)
#     ax1 = fig.add_subplot(111)
#     ax1.set_ylabel('somme des récompenses')
#     ax1.set_xlabel('épisode')
#     ax1.set_title('cartPole - v1')
#     ax1.plot(list_iteration, list_reward)
#     plt.show()
#
