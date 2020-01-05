import math
import random

import cv2
import gym
import keras
import matplotlib.pyplot as plt
import numpy
from gym import wrappers

## MEMO
# action : 2 valeurs (droite / gauche)
# environment : [position of cart, velocity of cart, angle of pole, rotation rate of pole]
# state / next_state = tableau de 4 elements
# action : 0 (rien), 1 (tirer), 2 (gauche) and 3 (droite)
# reward : 1 si la baton n'est pas tombé
# done : True ou False

# frame = image de 210×160 pixels avec une palette de 128 couleurs
from keras import Sequential, Input, Model
from keras.layers import Dense, Conv2D, Convolution2D, Lambda, Flatten, merge, Multiply
from keras.losses import huber_loss
from keras.optimizers import Adam, RMSprop
import keras.backend as K


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

        input_shape = (84, 84, 1)
        self.model = Sequential()
        self.model.add(Conv2D(32,
                              8,
                              strides=(4, 4),
                              padding="valid",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Conv2D(64,
                              4,
                              strides=(2, 2),
                              padding="valid",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Conv2D(64,
                              3,
                              strides=(1, 1),
                              padding="valid",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense((self.action_size,)))
        self.model.compile(loss="mean_squared_error",
                           optimizer=RMSprop(lr=0.00025,
                                             rho=0.95,
                                             epsilon=0.01),
                           metrics=["accuracy"])
        # We assume a theano backend here, so the "channels" are first.
        # ATARI_SHAPE = (4, 105, 80)
        #
        # # With the functional API we need to define the inputs.
        # frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
        # actions_input = keras.layers.Input((self.action_size,), name='mask')
        #
        # # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        # normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
        #
        # # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
        # conv_1 = keras.layers.convolutional.Convolution2D(
        #     16, 8, 8, subsample=(4, 4), activation='relu'
        # )(normalized)
        # # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
        # conv_2 = keras.layers.convolutional.Convolution2D(
        #     32, 4, 4, subsample=(2, 2), activation='relu'
        # )(conv_1)
        # # Flattening the second convolutional layer.
        # conv_flattened = keras.layers.core.Flatten()(conv_2)
        # # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        # hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        # # "The output layer is a fully-connected linear layer with a single output for each valid action."
        # output = keras.layers.Dense(self.action_size)(hidden)
        # # Finally, we multiply the output by the mask!
        # filtered_output = keras.layers.merge([output, actions_input], mode='mul')
        #
        # self.model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
        # optimizer = optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        # self.model.compile(optimizer, loss='mse')

        # if self.dueling:
        #     # Dueling Network
        #     # Q = Value of state + (Value of Action - Mean of all action value)
        #     hidden_feature_2 = Dense(512, activation='relu')(flat_feature)
        #     state_value_prediction = Dense(1)(hidden_feature_2)
        #     q_value_prediction = merge([q_value_prediction, state_value_prediction],
        #                                mode=lambda x: x[0] - K.mean(x[0]) + x[1],
        #                                output_shape=(self.num_actions,))

        # select_q_value_of_action = Multiply()([q_value_prediction, action_one_hot])
        # target_q_value = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True),)(select_q_value_of_action)
        # self.model = Model(inputs=[input_frame, action_one_hot], outputs=[q_value_prediction, target_q_value])

        # TODO: faire avec 3 réseaux convolutionnels

        # self.model = Model(input=[frames_input, actions_input], output=filtered_output)
        # optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        # self.model.compile(optimizer, loss=huber_loss)
        #
        # self.target_model = Model(input=[frames_input, actions_input], output=filtered_output)
        # optimizer2 = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        # self.target_model.compile(optimizer2, loss=huber_loss)


    def act(self, state, policy="greedy"):
        """
        Choisit une action pour l'état donné
        :param state: l'état courant de l'agent
        :param policy: la politique utilisée par l'agent
        :return: l'action choisie par la politique
        """

        if numpy.random.random() <= self.exploration_rate:
            return env.action_space.sample()
        return numpy.argmax(self.model.predict(state))
        # argmax retourne l'indice de la meilleure valeur
        # if policy == "greedy":
        #     if numpy.random.rand() <= self.exploration_rate:
        #         # on retourne une action aléatoire (exploration)
        #         return random.randrange(self.action_size)
        #     else:
        #         # on retourne la meilleure action prédite par le réseau (intensification)
        #         q_values = self.model.predict(state)
        #         # print(q_values)
        #         return numpy.argmax(q_values)
        # elif policy == "boltzmann":
        #     if numpy.random.rand() <= self.exploration_rate:
        #         # on retourne une action aléatoire (exploration)
        #         return random.randrange(self.action_size)
        #     else:
        #         tau = 0.8
        #         q_values = self.model.predict(state)
        #         sum_q_values = 0
        #         boltzmann_probabilities = [0 for _ in range(len(q_values[0]))]
        #         for i in range(len(q_values[0])):
        #             # calcul de la somme des exp(q_val / tau)
        #             sum_q_values += numpy.exp(q_values[0][i] / tau)
        #         for i in range(len(q_values[0])):
        #             # calcul de la probabilité de Boltzmann pour chaque action
        #             current_q_value = q_values[0][i]
        #             # sum(q_values[:i]) : les q_valeurs des actions entre 0 et i
        #             boltzmann_probabilities[i] = numpy.exp(current_q_value/tau) / sum_q_values
        #         # on retourne l'action qui a la plus grande probabilité
        #         return numpy.argmax(boltzmann_probabilities)
        # else:
        #     # la politique demandée n'est pas implémentée donc on retourne une action aléatoire
        #     return random.randrange(self.action_size)

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
        # mini_batch = self.memory.sample()
        # x_batch, y_batch = [], []
        # # on a assez d'experiences en memoire pour avoir un minibatch
        # if mini_batch is not None:
        #     for state, action, reward, next_state, done in mini_batch:
        #         losses = []
        #         if not done:
        #             # TODO: ici on utilise le target model pour la prédiction du prochain état pour plus de stabilité dans le réseau (évite de modifier "en double" vu que Q et Q^ sont modifiées toutes les deux)
        #             # q_value = (reward + self.gamma * numpy.amax(self.target_model.predict(next_state)[0]))
        #             q_value = (reward + self.gamma * numpy.amax(self.model.predict(next_state)[0]))
        #         else:
        #             q_value = reward
        #         q_values = self.model.predict(state)  # predictions pour un l'état donné en paramètre
        #         q_values[0][action] = q_value  # mise a jour de la Q-valeur de l'action (pour l'état)
        #         x_batch.append(state[0])
        #         y_batch.append(q_values[0])
        #         # TODO: utiliser la backpropagation
        #         # TODO: obligatoire pour que le réseau apprenne
        #         # TODO : lequel prédit sur target lequel sur model ?
        #         # q_value_previous = self.target_model.predict(state)[0]
        #         # erreur = carré de la différence entre l'état courant et l'état futur
        #         # loss = math.pow((q_value_previous - q_value), 2)
        #         # losses.append(loss)
        #         # loss.
        #
        #
        #         # entrainement sur le mini batch
        #         # if done:
        #         #     self.model.fit(state, q_values, verbose=0, batch_size=self.memory.batch_size)
        #         # else:
        #     self.model.fit(numpy.array(x_batch), numpy.array(y_batch), verbose=0, batch_size=self.memory.batch_size)
        #
        #     if self.exploration_rate > self.exploration_min:
        #         self.exploration_rate *= self.exploration_decay
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
    # test_preprocessing(0)  # TODO: à décommenter
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
        state = preprocessing(env.reset())
        # [ 0.0273208   0.01715898 -0.03423725  0.01013875] => [[ 0.0273208   0.01715898 -0.03423725  0.01013875]]
        # state = numpy.reshape(state, [1, env.observation_space.shape[0]])  # TODO: pour avoir un vecteur de 1
        steps = 1
        sum_reward = 0
        while True:
            action = agent.act(state, "greedy")  # choix d'une action (greedy: soit aléatoire soit via le réseau)
            next_state, reward, done, _ = env.step(action)  # on "exécute" l'action sur l'environnement
            # next_state = numpy.reshape(next_state, [1, env.observation_space.shape[0]])  # TODO:
            agent.remember(state, action, reward, next_state, done)
            state = preprocessing(next_state)
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
    # print("Meilleur reward obtenu", max(liste_rewards), "lors de l'épisode", liste_rewards.index(max(liste_rewards)))

    # gym.upload('./video', api_key='blah')
