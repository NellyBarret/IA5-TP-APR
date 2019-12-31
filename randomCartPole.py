import math
from collections import namedtuple
from random import random

import gym
import matplotlib.pyplot as plt
import torch.nn as nn


## MEMO
# action_sample : espace d'actions : 2 valuers (droite / gauche)


class RandomAgent:
    """
    Agent qui choisit des actions de manière aléatoire
    """
    def __init__(self, action_space):
        """
        Initialisation générale
        """
        self.action_space = action_space

    def act(self):
        """
        Choisit une action aléatoirement parmi l'espace d'actions
        """
        return self.action_space.sample()  # on retourne une action aléatoirement parmi les actions possibles


# 2.1 - question 2
def evolution_rewards(liste_rewards):
    """
    Trace l'évolution de la somme des récompenses par épisodes
    """
    plt.plot([i for i in range(len(liste_rewards))], liste_rewards)
    plt.title("Evolution de la somme des récompenses par épisodes")
    plt.xlabel('Nombre d\'épisodes')
    plt.ylabel('Somme des récompenses')
    plt.show()


# 2.1 - question 1
if __name__ == '__main__':
    env = gym.make("CartPole-v1")  # creation de l'environnement
    agent = RandomAgent(env.action_space)  # creation de l'agent

    nb_episodes = 100

    all_rewards = []
    done = False  # pour savoir quand on s'arrete (le baton est tombé ou il est sorti de l'environnement)

    for i in range(nb_episodes):
        total_reward = 0
        env.render()
        ob = env.reset()
        while True:
            action = agent.act()
            _, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break
        all_rewards.append(total_reward)
    evolution_rewards(all_rewards)  # courbe d'évolution de la somme des récompenses
    env.close()
