import math

import gym
import matplotlib.pyplot as plt
from gym import wrappers


## MEMO
# action : 2 valeurs (droite / gauche)
# environment : [position of cart, velocity of cart, angle of pole, rotation rate of pole]
# state / next_state = tableau de 4 elements
# action : 0 ou 1
# reward : 1 si la baton n'est pas tombé
# done : True ou False


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
    plt.plot([i for i in range(1, len(liste_rewards)+1)], liste_rewards)
    plt.title("Évolution de la somme des récompenses par épisode")
    plt.xlabel('Numéro de  l\'épisode')
    plt.ylabel('Somme des récompenses')
    plt.show()


# 2.1 - question 1
if __name__ == '__main__':
    env = gym.make("CartPole-v1")  # creation de l'environnement
    # env = wrappers.Monitor(env, './video')
    agent = RandomAgent(env.action_space)  # creation de l'agent
    liste_rewards = []

    nb_episodes = 1000
    for i in range(nb_episodes):
        total_reward = 0
        # env.render()
        env.reset()
        # state = env.reset()
        while True:
            # env.render()
            action = agent.act()
            _, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break
        liste_rewards.append(total_reward)
    evolution_rewards(liste_rewards)  # courbe d'évolution de la somme des récompenses
    print("Meilleure récompense obtenue", max(liste_rewards), "lors de l'épisode", liste_rewards.index(max(liste_rewards)))
    env.close()
    # gym.upload('./video', api_key='blah')
