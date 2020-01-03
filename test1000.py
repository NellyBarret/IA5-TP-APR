import gym
import numpy
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow import keras
import random
from pprint import pprint


class Memory:
    """
    Classe représentant la mémoire de l'agent (utile pour l'expérience replay)
    """
    def __init__(self, max_size, batch_size):
        """
        Initialise la mémoire de l'agent
        @param max_size: taille maximale de la mémoire (par défaut 100 000)
        @param batch_size: taille du batch généré sur la mémoire de l'agent
        """
        self.max_size = max_size
        self.memory = [[] for _ in range(self.max_size)]  # bien penser a initialiser la memoire pour ne pas avoir d'index out of range
        self.position = 0
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        Ajoute une expérience à la mémoire de l'agent
        @param state: l'état courant de l'agent
        @param action: l'action choisie par l'agent
        @param reward: la récompense gagnée
        @param next_state: l'état d'arrivée après exécution de l'action
        @param done: True si l'expérience est finie (la bâton est tombé ou l'agent est sorti de l'environnement)
        """
        # on ajoute l'experience et on incremente la position dans la memoire
        self.memory[self.position] = [state, action, reward, next_state, done]
        self.position = (self.position + 1) % self.max_size  # modulo la taille max pour ne pas depasser

    def sample(self):
        """
        Construit un batch aléatoire sur la mémoire de l'agent
        """
        if (sum(len(item) > 0 for item in self.memory) < self.batch_size) or [] in self.memory:
            # pas assez d'experiences pour construire le batch ou il existe des expériences vides
            # comme sample prend des éléments aléatoirement, on vérifie qu'il n'y a pas d'éléméents vides (sinon unpack) TODO: mieux expliquer
            return None
        else:
            # creation du batch aleatoire parmi les elements de la memoire
            batch = random.sample(self.memory, self.batch_size)
            return batch


class DQNAgent:
    """
    Classe représentant l'agent DQN et son réseau
    """
    def __init__(self, params):
        """
        Initialise le réseau et l'agent
        @param params: le dictionnaire contenant les paramètres du réseau et de l'agent
        """
        self.state_size = params['state_size']  # taille de l'entrée du réseau
        self.action_size = params['action_size']  # taille de sortie du réseau

        self.memory = Memory(params['memory_size'], params['batch_size'])  # m&moire pour l'expérience replay

        self.gamma = params['gamma']
        self.learning_rate = params['learning_rate']
        self.exploration_rate = params['exploration_rate']  # greedy
        self.exploration_decay = params['exploration_decay']
        self.exploration_min = params['exploration_min']

        # model "de base"
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(self.state_size,), activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        # target model pour la stabilité
        self.target_model = Sequential()
        self.target_model.add(Dense(24, input_shape=(self.state_size,), activation='relu'))
        self.target_model.add(Dense(24, activation='relu'))
        self.target_model.add(Dense(self.action_size, activation='linear'))
        self.target_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def act(self, state):
        """
        Choisit une action pour l'état donné
        @param state: l'état courant de l'agent
        """
        if numpy.random.rand() <= self.exploration_rate:
            # on retourne une action aléatoire (exploration)
            return random.randrange(self.action_size)
        else:
            # on retourne la meilleure action prédite par le réseau (intensification)
            q_values = self.model.predict(state)
            # print(q_values)
            return numpy.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """
        Ajoute une expérience à la mémoire de l'agent
        @param state: l'état courant de l'agent
        @param action: l'action choisie par l'agent
        @param reward: la récompense gagnée
        @param next_state: l'état d'arrivée après exécution de l'action
        @param done: True si l'expérience est finie (la bâton est tombé ou l'agent est sorti de l'environnement)
        """
        self.memory.add(state, action, reward, next_state, done)

    def experience_replay(self, done):
        """
        @param back_propagation: valeur de la rétropropagation (à donner à la fonction fit)
        """
        mini_batch = self.memory.sample()
        # on a assez d'experiences en memoire pour avoir un minibatch
        if mini_batch is not None:
            for state, action, reward, next_state, done in mini_batch:
                if not done:
                    # TODO: ici on utilsie le target model pour la prédiction du prochain état pour plus de stabilité dans le réseau (évite de modifier "en double" vu que Q et Q^ sont modifiées toutes les deux)
                    q_value = (reward + self.gamma * numpy.amax(self.target_model.predict(next_state)[0]))
                    # q_value = (reward + self.gamma * numpy.amax(self.model.predict(next_state)[0]))
                else:
                    q_value = reward
                q_values = self.model.predict(state)  # predictions pour un l'état donné en paramètre
                q_values[0][action] = q_value  # mise a jour de la Q-valeur de l'action (poue l'état)
                # entrainement sur le mini batch
                if done:
                    self.model.fit(state, q_values, verbose=0)
                else:
                    self.model.fit(state, q_values, verbose=0)
                # TODO: utiliser la backpropagation
            self.exploration_rate *= self.exploration_decay
            self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    def update_target_network(self):
        """
        Mise à jour du target model à partir du model
        """
        self.target_model.set_weights(self.model.get_weights())


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    # constantes pour l'agent DQN
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    memory_size = 10
    batch_size = 3
    gamma = 0.95
    learning_rate = 0.001
    exploration_rate = 1
    exploration_decay = 0.995
    exploration_min = 0.01

    # constantes pour l'exécution
    nb_episodes = 20
    update_target_network = 20

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
    agent = DQNAgent(params)

    for i in range(nb_episodes):
        state = env.reset()
        # [ 0.0273208   0.01715898 -0.03423725  0.01013875] => [[ 0.0273208   0.01715898 -0.03423725  0.01013875]]
        state = numpy.reshape(state, [1, env.observation_space.shape[0]])  # TODO: pour avoir un vecteur de 1
        steps = 1
        sum_reward = 0
        while True:
            action = agent.act(state)  # choix d'une action (greedy: soit aléatoire soit via le réseau)
            next_state, reward, done, _ = env.step(action)  # on "exécute" l'action sur l'environnement
            next_state = numpy.reshape(next_state, [1, env.observation_space.shape[0]])  # TODO:
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            sum_reward += reward
            agent.experience_replay(done)
            if done:
                print("epsiode", i, "- steps : ", steps,"- somme reward", sum_reward)
                break
            if steps % update_target_network == 0:
                # on met à jour le target network tous les `update_target_network` pas
                print("the target network is updating")
                agent.update_target_network()
            steps += 1
