import math
import random

import cv2
import gym
import matplotlib.pyplot as plt
import numpy
from gym import wrappers


## MEMO
# action : 2 valeurs (droite / gauche)
# environment : [position of cart, velocity of cart, angle of pole, rotation rate of pole]
# state / next_state = tableau de 4 elements
# action : 0 ou 1
# reward : 1 si la baton n'est pas tombé
# done : True ou False

# frame = image de 210×160 pixels avec une palette de 128 couleurs
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Memory:
    """
    Classe représentant la mémoire de l'agent (utile pour l'expérience replay)
    """
    def __init__(self, max_size, batch_size):
        """
        Initialise la mémoire de l'agent
        :param max_size: taille maximale de la mémoire (par défaut 100 000)
        :param batch_size: taille du batch généré sur la mémoire de l'agent
        """
        self.max_size = max_size
        self.memory = [[] for _ in range(self.max_size)]  # penser a initialiser pour ne pas avoir d'index out of range
        self.position = 0
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        Ajoute une expérience à la mémoire de l'agent
        :param state: l'état courant de l'agent
        :param action: l'action choisie par l'agent
        :param reward: la récompense gagnée
        :param next_state: l'état d'arrivée après exécution de l'action
        :param done: True si l'expérience est finie (la bâton est tombé ou l'agent est sorti de l'environnement)
        """
        # on ajoute l'experience et on incremente la position dans la memoire
        self.memory[self.position] = [state, action, reward, next_state, done]
        self.position = (self.position + 1) % self.max_size  # modulo la taille max pour ne pas depasser

    def sample(self):
        """
        Construit un batch aléatoire sur la mémoire de l'agent
        :return: le batch
        """
        if (sum(len(item) > 0 for item in self.memory) < self.batch_size) or [] in self.memory:
            # pas assez d'experiences pour construire le batch ou il existe des expériences vides
            # comme sample prend des éléments aléatoirement, on vérifie qu'il n'y a pas d'éléméents vides (sinon unpack)
            # TODO: mieux expliquer
            return None
        else:
            # creation du batch aleatoire parmi les elements de la memoire
            batch = random.sample(self.memory, self.batch_size)
            return batch

    def __len__(self):
        """
        Retourne le nombre d'élélemnts (non nuls) dans la mémoire
        :return: le nombre d'éléments dans la mémoire
        """
        return sum(len(item) > 0 for item in self.memory)  # len(self.memory)


class BreakoutAgent:
    """
    Classe représentant l'agent du Breakout et son réseau
    """
    def __init__(self, params):
        """
        Initialise le réseau et l'agent
        :param params: le dictionnaire contenant les paramètres du réseau et de l'agent
        """
        self.state_size = params['state_size']  # taille de l'entrée du réseau
        self.action_size = params['action_size']  # taille de sortie du réseau

        self.memory = Memory(params['memory_size'], params['batch_size'])  # deque(maxlen=100000) -- mémoire pour l'expérience replay

        self.gamma = params['gamma']
        self.learning_rate = params['learning_rate']
        self.exploration_rate = params['exploration_rate']  # greedy
        self.exploration_decay = params['exploration_decay']
        self.exploration_min = params['exploration_min']

        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state, policy="greedy"):
        """
        Choisit une action pour l'état donné
        :param state: l'état courant de l'agent
        :param policy: la politique utilisée par l'agent
        :return: l'action choisie par la politique
        """
        # argmax retourne l'indice de la meilleure valeur
        if policy == "greedy":
            if numpy.random.rand() <= self.exploration_rate:
                # on retourne une action aléatoire (exploration)
                return random.randrange(self.action_size)
            else:
                # on retourne la meilleure action prédite par le réseau (intensification)
                q_values = self.model.predict(state)
                return numpy.argmax(q_values)
        elif policy == "boltzmann":
            if numpy.random.rand() <= self.exploration_rate:
                # on retourne une action aléatoire (exploration)
                return random.randrange(self.action_size)
            else:
                tau = 0.8
                q_values = self.model.predict(state)
                sum_q_values = 0
                boltzmann_probabilities = [0 for _ in range(len(q_values[0]))]
                for i in range(len(q_values[0])):
                    # calcul de la somme des exp(q_val / tau)
                    sum_q_values += numpy.exp(q_values[0][i] / tau)
                for i in range(len(q_values[0])):
                    # calcul de la probabilité de Boltzmann pour chaque action
                    current_q_value = q_values[0][i]
                    # sum(q_values[:i]) : les q_valeurs des actions entre 0 et i
                    boltzmann_probabilities[i] = numpy.exp(current_q_value/tau) / sum_q_values
                # on retourne l'action qui a la plus grande probabilité
                return numpy.argmax(boltzmann_probabilities)
        else:
            # la politique demandée n'est pas implémentée donc on retourne une action aléatoire
            return random.randrange(self.action_size)

    def remember(self, state, action, reward, next_state, done):
        """
        Ajoute une expérience à la mémoire de l'agent
        :param state: l'état courant de l'agent
        :param action: l'action choisie par l'agent
        :param reward: la récompense gagnée
        :param next_state: l'état d'arrivée après exécution de l'action
        :param done: True si l'expérience est finie (la bâton est tombé ou l'agent est sorti de l'environnement)
        """""
        self.memory.add(state, action, reward, next_state, done)
        # self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        """
        Calcule les prédictions, met à jour le modèle et entraine le réseau
        La rétropropagation est faite par la fonction fit
        """
        x_batch, y_batch = [], []
        minibatch = self.memory.sample()
        if minibatch is not None:
            for state, action, reward, next_state, done in minibatch:
                y_target = self.model.predict(state)
                if done:
                    y_target[0][action] = reward
                else:
                    y_target[0][action] = reward + self.gamma * numpy.max(self.target_model.predict(next_state)[0])
                x_batch.append(state[0])
                y_batch.append(y_target[0])

            self.model.fit(numpy.array(x_batch), numpy.array(y_batch), batch_size=len(x_batch), verbose=0)
            if self.exploration_rate > self.exploration_min:
                self.exploration_rate *= self.exploration_decay

    def update_target_network(self):
        """
        Met à jour le target model à partir du model
        """
        self.target_model.set_weights(self.model.get_weights())


# 3 - question 1
def preprocessing(observation):
    """
    Preprocessing des images (state) : réduction de la dimension (210*160*3 => 84*84*1) + noir et blanc
    """
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return numpy.reshape(observation, (84, 84, 1))


def test_preprocessing(action):
    env.reset()
    state, reward, done, _ = env.step(action)
    print("Before processing: " + str(numpy.array(state).shape))
    state = preprocessing(state)
    print("After processing: " + str(numpy.array(state).shape))


if __name__ == '__main__':

    env = gym.make("BreakoutNoFrameskip-v4")  # creation de l'environnement
    env = gym.wrappers.Monitor(env, "recordingDQNbreakout", force=True)
    test_preprocessing(0)  # TODO: à décommenter
    # env = wrappers.Monitor(env, './video')
    # constantes pour l'agent DQN
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    memory_size = 100000
    batch_size = 20  # 64
    gamma = 0.95  # 0.99
    learning_rate = 0.001
    exploration_rate = 1
    exploration_decay = 0.995
    exploration_min = 0.01

    # constantes pour l'exécution
    nb_episodes = 200
    update_target_network = 100

    # creation de l'agent avec ses paramètres
    params = {
        'state_size': state_size,
        'action_size': action_size,
        'memory_size': memory_size,
        'batch_size': batch_size,
        'gamma': gamma,
        'learning_rate': learning_rate,
        'exploration_rate': exploration_rate,
        'exploration_decay': exploration_decay,
        'exploration_min': exploration_min

    }
    agent = BreakoutAgent(params)  # creation de l'agent

    liste_rewards = []
    done = False  # pour savoir quand on s'arrete (le baton est tombé ou il est sorti de l'environnement)

    for i in range(nb_episodes):
        state = env.reset()
        # [ 0.0273208   0.01715898 -0.03423725  0.01013875] => [[ 0.0273208   0.01715898 -0.03423725  0.01013875]]
        # state = numpy.reshape(state, [1, env.observation_space.shape[0]])  # TODO: pour avoir un vecteur de 1
        steps = 1
        sum_reward = 0
        while True:
            action = agent.act(state, "greedy")  # choix d'une action (greedy: soit aléatoire soit via le réseau)
            next_state, reward, done, _ = env.step(action)  # on "exécute" l'action sur l'environnement
            # next_state = numpy.reshape(next_state, [1, env.observation_space.shape[0]])  # TODO:
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            sum_reward += reward
            agent.experience_replay()
            if done:
                print("epsiode", i, "- steps : ", steps, "- somme reward", sum_reward)
                break
            # if steps % update_target_network == 0:
            #     # on met à jour le target network tous les `update_target_network` pas
            #     print("the target network is updating")
            #     agent.update_target_network()
            steps += 1
        liste_rewards.append(sum_reward)
    print("Meilleur reward obtenu", max(liste_rewards), "lors de l'épisode", liste_rewards.index(max(liste_rewards)))

    # gym.upload('./video', api_key='blah')
