import math
from collections import namedtuple
import random
import numpy
import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from keras import Sequential

GAMMA = 0.95
BATCH_SIZE = 20



EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

## MEMO
# action : 2 valeurs (droite / gauche)
# environment : [position of cart, velocity of cart, angle of pole, rotation rate of pole]
# state / next_state = tableau de 4 elements
# action : 0 ou 1
# reward : 1 si la baton n'est pas tombé
# done : True ou False
# policy: strategie pour choisir la meilleure action pour un état donné (celle qui permet de maximiser le gain)
from tensorflow_core.python.keras.layers import Dense
from torch.optim import Adam


# class DQNAgent:
#     """
#     Agent qui choisit des actions de manière aléatoire
#     """
#     def __init__(self, observation_space, action_space, epsilon, epsilon_maj, epsilon_min, gamma, batch_size):
#         """
#         Initialisation générale
#         @param observation_space: TODO
#         @param action_space: espace des actions possibles
#         @param epsilon: taux d'exploration
#         @param batch_size: TODO
#         """
#         self.action_space = action_space
#         self.memory = []
#         self.epsilon = epsilon
#         self.epsilon_maj = epsilon_maj
#         self.epsilon_min = epsilon_min
#         self.gamma = gamma
#         self.batch_size = batch_size
#         self.model = nn.Sequential()
#         self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
#         self.model.add(Dense(24, activation="relu"))
#         self.model.add(Dense(self.action_space, activation="linear"))
#         self.model.compile(loss="mse", optimizer=Adam(lr=self.epsilon))
#
#     def remember(self, state, action, reward, next_state, done):
#         """
#         Ajoute une interaction à la mémoire de l'agent
#         @param state: etat courant
#         @param action: action effectuee
#         @param reward: recompense recue de l'environnement
#         @param next_state: etat dans lequel on arrive
#         @param done: pour arreter l'agent quand il a fini
#         """
#         self.memory.append([state, action, reward, next_state, done])
#
#     def act(self, state):
#         """
#         Choisit la meilleure action
#         @param state: l'etat courant de l'agent
#         """
#         rand = random.random()
#         if rand < self.epsilon:
#             return random.randrange(self.action_space)
#         q_valeurs = self.model.predict(state)
#         return torch.argmax(q_valeurs[0])
#
#     def experience_replay(self):
#         if len(self.memory) < self.batch_size:
#             return
#         batch = random.sample(self.memory, self.batch_size)  # creation du batch
#         for state, action, reward, state_next, done in batch:
#             q_valeurs_maj = reward
#             if not done:
#                 q_valeurs_maj = (reward + self.gamma * torch.argmax(self.model.predict(state_next)[0]))
#             q_valeurs = self.model.predict(state)
#             q_valeurs[0][action] = q_valeurs_maj
#             self.model.fit(state, q_valeurs)
#         self.epsilon *= self.epsilon_maj
#         self.epsilon = max(self.epsilon_min, self.epsilon)


class DQNAgent:
    # def __init__(self, observation_space, action_space):  # input_size, output_size,
    #     self.input_size = 4  # input_size
    #     self.output_size = 2  # output_size
    #     self.model = Sequential()
    #     self.model.add(Dense(30, input_shape=(self.input_size,), activation="relu"))
    #     self.model.add(Dense(30, activation="relu"))
    #     self.model.add(Dense(self.output_size, activation="linear"))
    #     self.model.compile(loss="mse", optimizer=Adam(lr=self.epsilon))
    #     #
    #     # self.model = nn.Sequential(nn.Linear(self.input_size, 30),
    #     #               nn.ReLU(),
    #     #               nn.Linear(30, 30),
    #     #               nn.ReLU(),
    #     #               nn.Linear(30, self.output_size))
    #     self.memory = []
    #     self.memory_max_size = 100000
    #     self.action_space = action_space
    #     self.observation_space = observation_space
    #     self.exploration_rate = EXPLORATION_MAX
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))

        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss='mse', optimizer=self.optimizer)

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
        if numpy.random.rand() < self.exploration_rate:
            # on prend une action aléatoirement parmi les actions possibles
            return random.randrange(self.action_space)
        # on predit les q valeur pour chaque action de l'etat
        q_values = self.model.predict(state)
        # on renvoie la meilleure valeur (donc la meilleure action)
        return numpy.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)  # creation du batch
        for state, action, reward, state_next, done in batch:
            q_valeurs_maj = reward
            if not done:
                q_valeurs_maj = (reward + GAMMA * torch.argmax(self.model.predict(state_next)[0]))
            q_valeurs = self.model.predict(state)
            q_valeurs[0][action] = q_valeurs_maj
            self.model.fit(state, q_valeurs)
        self.epsilon *= EXPLORATION_DECAY
        self.epsilon = max(EXPLORATION_MIN, self.epsilon)


# 2.3 - question 5
if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNAgent(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = numpy.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            # env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = numpy.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print
                "Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step)
                break
            dqn_solver.experience_replay()

    # env = gym.make('CartPole-v1')  # creation de l'environnement
    # # env = gym.make('FrozenLake8x8-v0')
    # nb_episodes = 5000  # 100
    # # TODO: regarder pour env.observation_space.n
    # q_table = numpy.zeros([env.observation_space.n, env.action_space.n])
    # alpha = 0.1
    # gamma = 0.9  # 0.6
    # epsilon = 0.1
    # eta = 0.628
    #
    # for i in range(nb_episodes):
    #     state = env.reset()
    #     while True:
    #         action = numpy.argmax(q_table[state, :] + numpy.random.randn(1, env.action_space.n)*(1./(i+1)))
    #         next_state, reward, done, _ = env.step(action)
    #         # mise a jour des q valeurs avec l'equation de bellman
    #         q_table[state, action] += eta * (reward + gamma*numpy.max(q_table[next_state, :]) - q_table[state, action])
    #         state = next_state
    #         if done:
    #             break
    # print(q_table)

    # 2.3 - question 6
    # if random.uniform(0, 1) < epsilon:
    #     # exploration
    #     action = env.action_space.sample()  # action aleatoirement choisie
    # else:
    #     action = numpy.argmax(q_table[state])

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
