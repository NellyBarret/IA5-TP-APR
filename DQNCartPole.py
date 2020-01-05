import gym
import numpy
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
from collections import deque
import matplotlib.pyplot as plt


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
        self.memory = [[] for _ in range(self.max_size)]  # penser a initialiser pour ne pas avoir d'index out of range
        self.position = 0
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        Ajoute une expérience à la mémoire de l'agent
        @param state: l'état courant de l'agent
        @param action: l'action choisie par l'agent
        @param reward: la récompense gagnée
        @param next_state: l'état d'arrivée après exécution de l'action
        @param done: True si l'expérience est finie (le bâton est tombé ou l'agent est sorti de l'environnement)
        """
        # on ajoute l'experience et on incrémente la position dans la mémoire
        self.memory[self.position] = [state, action, reward, next_state, done]
        self.position = (self.position + 1) % self.max_size  # modulo la taille max pour ne pas depasser

    def sample(self):
        """
        Construit un batch aléatoire sur la mémoire de l'agent
        :return: le batch d'expériences
        """
        if self.__len__() < self.batch_size:  # or [] in self.memory:
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
        Retourne le nombre d'éléments (non nuls) dans la mémoire
        :return: le nombre d'éléments dans la mémoire
        """
        return sum(len(item) > 0 for item in self.memory)  # len(self.memory)


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

        self.memory = deque(maxlen=params['memory_size'])  # Memory(params['memory_size'], params['batch_size'])  # deque(maxlen=100000) -- mémoire pour l'expérience replay
        self.batch_size = params['batch_size']

        self.gamma = params['gamma']
        self.learning_rate = params['learning_rate']
        self.exploration_rate = params['exploration_rate']  # greedy
        self.exploration_decay = params['exploration_decay']
        self.exploration_min = params['exploration_min']

        # model "de base"
        # self.model = nn.Sequential(
        #     nn.Linear(self.observation_space.shape[0], 30),
        #     nn.ReLU(),
        #     nn.Linear(30, 30),
        #     nn.ReLU(),
        #     nn.Linear(30, self.action_space.n)
        # )
        self.model = self.build_model()
        self.target_model = self.build_model()
        # self.model = Sequential()
        # self.model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        # self.model.add(Dense(48, activation='tanh'))
        # self.model.add(Dense(self.action_size, activation='linear'))
        # self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=self.exploration_decay))
        #
        # self.target_model = Sequential()
        # self.target_model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        # self.target_model.add(Dense(48, activation='tanh'))
        # self.target_model.add(Dense(self.action_size, activation='linear'))
        # self.target_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=self.exploration_decay))

        # self.model = Sequential(nn.Linear(self.state_size, 30),
        #               nn.ReLU(),
        #               nn.Linear(30, 30),
        #               nn.ReLU(),
        #               nn.Linear(30, self.action_size))
        # target model pour la stabilité

    def build_model(self):
        """
        Construit le modèle neuronal
        """
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state, policy="greedy"):
        """
        Choisit une action pour l'état donné
        @param state: l'état courant de l'agent
        @param policy: la politique utilisée par l'agent
        """
        # argmax retourne l'indice de la maeilleure valeur
        if policy == "greedy":
            if numpy.random.rand() < self.exploration_rate:
                # on retourne une action aléatoire (exploration)
                # return env.action_space.sample()
                return random.randrange(self.action_size)
            else:
                # on retourne la meilleure action prédite par le réseau (intensification)
                q_values = self.model.predict(state)
                # print(q_values)
                return numpy.argmax(q_values[0])
        elif policy == "boltzmann":
            if numpy.random.rand() <= self.exploration_rate:
                # on retourne une action aléatoire (exploration)
                return env.action_space.sample()
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
            # return random.randrange(self.action_size)
            return env.action_space.sample()

    def remember(self, state, action, reward, next_state, done):
        """
        Ajoute une expérience à la mémoire de l'agent
        @param state: l'état courant de l'agent
        @param action: l'action choisie par l'agent
        @param reward: la récompense gagnée
        @param next_state: l'état d'arrivée après exécution de l'action
        @param done: True si l'expérience est finie (la bâton est tombé ou l'agent est sorti de l'environnement)
        """
        # self.memory.add(state, action, reward, next_state, done)
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        """
        Calcule les prédictions, met à jour le modèle et entraine le réseau
        La rétropropagation est faite par la fonction fit
        """
        # states, q_val = [], []
        # batch = self.memory.sample()  # création du batch à partir de la mémoire de l'agent
        # if batch is not None:
        #     for state, action, reward, next_state, done in batch:
        #         # prédictions des q-valeurs pour toutes les actions de l'état
        #         q_values = self.model.predict(state)
        #         # mise a jour de la Q-valeur de l'action de l'état
        #         if done:
        #             q_values[0][action] = reward
        #         else:
        #             q_values[0][action] = reward + self.gamma * numpy.max(self.target_model.predict(next_state)[0])
        #         states.append(state[0]) # contient tous les états
        #         q_val.append(q_values[0])  # contient les prédictions des q_valeurs
        #         # TODO: a quel endroit ?
        #         # self.model.fit(state, q_values, batch_size=len(states), verbose=0)
        #         pred = self.model.predict(state)
        #         y = reward + self.gamma * numpy.max(self.target_model.predict(next_state)[0])
        #         self.model.fit(pred, y, batch_size=self.state_size, verbose=0)
        #     # mise à jour du réseau sur le batch
        #     # self.model.fit(numpy.array(states), numpy.array(q_val), batch_size=len(states), verbose=0)
        #     if self.exploration_rate > self.exploration_min:
        #         self.exploration_rate *= self.exploration_decay
        if len(self.memory) < self.batch_size:  # self.memory.batch_size:
            return
        # batch = self.memory.sample()
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.gamma * numpy.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    def update_target_network(self):
        """
        Met à jour le target model à partir du model
        """
        self.target_model.set_weights(self.model.get_weights())


def evolution_rewards(liste_rewards):
    """
    Trace l'évolution de la somme des récompenses par épisodes
    """
    plt.plot([i for i in range(len(liste_rewards))], liste_rewards)
    plt.title("Evolution de la somme des récompenses par épisodes")
    plt.xlabel('Nombre d\'épisodes')
    plt.ylabel('Somme des récompenses')
    plt.show()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    # constantes pour l'agent DQN
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    memory_size = 100000
    batch_size = 64  # 64
    gamma = 0.99  # 0.99  # importance des récompenses à l'infini
    learning_rate = 0.001  # taux d'apprentissage de l'erreur entre la cible et la prédiction
    exploration_rate = 1  # pour savoir si on prend une action random ou la meilleure action
    exploration_decay = 0.995  # pour faire descendre l'exploration_rate pour baisser le nombre d'explorations au fur et à mesure que l'agent apprend et devient meilleur
    exploration_min = 0.01

    # constantes pour l'exécution
    nb_episodes = 200
    update_target_network = 100  # pas pour mettre à jour le target network
    save_weights = False  # True pour sauvegarder les poids du réseau dans un fichier tous les save_step episodes
    save_step = 10  # pas pour sauvegarder les poids du réseau

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
    liste_rewards = []  # liste des récompenses obtenues pour chaque épisode, permet de tracer le plot
    global_counter = 0
    for i in range(nb_episodes):
        state = env.reset()
        # [ 0.0273208   0.01715898 -0.03423725  0.01013875] => [[ 0.0273208   0.01715898 -0.03423725  0.01013875]]
        state = numpy.reshape(state, [1, env.observation_space.shape[0]])  # TODO: pour avoir un vecteur de 1
        steps = 1
        sum_reward = 0
        while True:
            action = agent.act(state, "greedy")  # choix d'une action (greedy: soit aléatoire soit via le réseau)
            next_state, reward, done, _ = env.step(action)  # on "exécute" l'action sur l'environnement
            next_state = numpy.reshape(next_state, [1, env.observation_space.shape[0]])  # TODO:
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            sum_reward += reward
            agent.experience_replay()
            if done:
                print("Episode", i, "- nombre de pas : ", steps, "- somme récompenses", sum_reward)
                break
            if global_counter % update_target_network == 0:
                # on met à jour le target network tous les `update_target_network` pas
                print("Le target network se met à jour")
                agent.update_target_network()
            steps += 1
            global_counter += 1
        liste_rewards.append(sum_reward)
        if save_weights and i % save_step == 0:
            print("Sauvegarde des poids du modèle")
            agent.model.save_weights("./cartpole_dqn.h5")
    evolution_rewards(liste_rewards)
    print("Meilleure récompense obtenue", max(liste_rewards), "lors de l'épisode", liste_rewards.index(max(liste_rewards)))
