import torch
from torch import nn
from torch.nn import Sequential
import random
import numpy
import gym


class DQNAgent:
    def __init__(self, taille_etat, epsilon_diminution, taux_apprentissage, taille_batch):
        self.taille_etat = taille_etat  # observation_space.shape[0]
        self.taille_action = 4
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_diminution = epsilon_diminution
        self.taux_apprentissage = taux_apprentissage
        self.memoire = []
        self.taille_batch = taille_batch
        self.model = Sequential(
            nn.Linear(self.taille_etat, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, self.taille_action)  # self.action_space.n
        )
        self.fonction_erreur = nn.MSELoss()
        self.optimiseur = torch.optim.Adam()

    def maj_epsilon(self):
        self.epsilon *= self.epsilon_diminution

    def choisir_meilleure_action(self, etat):
        if random.random() <= self.epsilon:
            # choix aleatoire d'une action
            return random.randrange(self.taille_action)
        else:
            # prediction de la recompense pour l'etat
            prediction = self.model(etat)
            action = numpy.argmax(prediction) # on choisit l'action qui aura la meilleure recompense
            return action

    def ajouter_a_memoire(self, etat, action, recompense, prochain_etat, done):
        self.memoire.append([etat, action, recompense, prochain_etat, done])

    def experience_replay(self, taille_batch):
        taille_batch = min(taille_batch, len(self.memoire))
        mini_batch = random.sample(self.memoire, taille_batch)  # TODO
        entrees = numpy.zeros((taille_batch, self.taille_etat))
        sorties = numpy.zeros((taille_batch, self.taille_action))

        for i, (etat, action, recompense, prochain_etat, done) in enumerate(mini_batch):
            cible = self.model(etat)
            if done:
                cible[action] = recompense
            else:
                cible[action] = recompense + self.gamma * numpy.max(self.model(prochain_etat)) # equation de bellman
            entrees[i] = etat
            sorties[i] = cible
        return self.model.fit(entrees, sorties)

if __name__ == '__main__':
    nb_episodes = 100
    env = gym.make('CartPole-v1')
    agent = DQNAgent(4, 0.999995, 0.001, 32)
    for i in range(nb_episodes):
        etat = env.reset()
        while True:
            action = agent.choisir_meilleure_action(etat)
            agent.maj_epsilon()
            prochain_etat, recompense, done, _ = agent.act()
