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
    plt.title("Evolution de la somme des récompenses par épisodes")
    plt.xlabel('Nombre d\'épisodes')
    plt.ylabel('Somme des récompenses')
    plt.show()


# 2.1 - question 1
if __name__ == '__main__':
    env = gym.make("CartPole-v1")  # creation de l'environnement
    # env = wrappers.Monitor(env, './video')
    agent = RandomAgent(env.action_space)  # creation de l'agent

    nb_episodes = 100

    all_rewards = []
    done = False  # pour savoir quand on s'arrete (le baton est tombé ou il est sorti de l'environnement)

    for i in range(nb_episodes):
        total_reward = 0
        # env.render()
        state = env.reset()
        while True:
            # env.render()
            action = agent.act()
            _, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break
        all_rewards.append(total_reward)
    evolution_rewards(all_rewards)  # courbe d'évolution de la somme des récompenses
    print("Meilleur reward obtenu:", max(all_rewards), "lors de l'épisode", all_rewards.index(max(all_rewards)))
    env.close()
    # gym.upload('./video', api_key='blah')
