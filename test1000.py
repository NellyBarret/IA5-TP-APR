import gym
import numpy
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow import keras
import random
from pprint import pprint


class Memory:
    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.memory = [[] for _ in range(self.max_size)]  # bien penser a initialiser la memoire pour ne pas avoir d'index out of range
        self.position = 0
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        # on ajoute l'experience et on incremente la position dans la memoire
        self.memory[self.position] = [state, action, reward, next_state, done]
        self.position = (self.position + 1) % self.max_size  # modulo la taille max pour ne pas depasser

    def sample(self):
        if (sum(len(item) > 0 for item in self.memory) < self.batch_size) or [] in self.memory:
            # pas assez d'experiences pour construire le batch ou il existe des expériences vides
            # comme sample prend des éléments aléatoirement, on vérifie qu'il n'y a pas d'éléméents vides (sinon unpack) TODO: mieux expliquer
            return None
        else:
            # creation du batch aleatoire parmi les elements de la memoire
            batch = random.sample(self.memory, self.batch_size)
            return batch


class DQNAgent:
    def __init__(self, state_size, action_size, memory_size, batch_size, gamma, learning_rate, exploration_rate, exploration_decay, exploration_min):
        self.state_size = state_size  # taille de l'entrée du réseau
        self.action_size = action_size  # taille de sortie du réseau

        self.memory = Memory(memory_size, batch_size)  # m&moire pour l'expérience replay

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate  # greedy
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(self.state_size,), activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def act(self, state):
        if numpy.random.rand() <= self.exploration_rate:
            # on retourne une action aléatoire (exploration)
            return random.randrange(self.action_size)
        else:
            # on retourne la meilleure action prédite par le réseau (intensification)
            q_values = self.model.predict(state)
            # print(q_values)
            return numpy.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def experience_replay(self):
        mini_batch = self.memory.sample()
        # on a assez d'experiences en memoire pour avoir un minibatch
        if mini_batch is not None:
            for state, action, reward, next_state, done in mini_batch:
                q_update = reward
                if not done:
                    # print(self.model.predict(next_state))
                    q_update = (reward + self.gamma * numpy.amax(self.model.predict(next_state)[0]))
                q_values = self.model.predict(state)
                q_values[0][action] = q_update
                # entrainement sur le mini batch
                self.model.fit(state, q_values, verbose=0)
            self.exploration_rate *= self.exploration_decay
            self.exploration_rate = max(self.exploration_min, self.exploration_rate)


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
    nb_episodes = 10

    # creation de l'agent
    agent = DQNAgent(
        state_size, action_size,
        memory_size,
        batch_size,
        gamma,
        learning_rate,
        exploration_rate, exploration_decay, exploration_min
    )

    for i in range(nb_episodes):
        state = env.reset()
        state = numpy.reshape(state, [1, env.observation_space.shape[0]])  # TODO: pour avoir un vecteur de 1
        score = 0
        sum_reward = 0
        while True:
            score += 1
            action = agent.act(state)  # choix d'une action (greedy: soit aléatoire soit via le réseau)
            next_state, reward, done, _ = env.step(action)  # on "exécute" l'action sur l'environnement
            reward = reward if not done else -reward  # TODO:
            next_state = numpy.reshape(next_state, [1, env.observation_space.shape[0]])  # TODO:
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            sum_reward += reward
            if done:
                print("epsiode", i, "- score : ", score,"- somme reward", sum_reward)
                break
            agent.experience_replay()
