import random
import gym

# MEMO
# action_sample : espace d'actions : 2 valeurs (droite / gauche)
# state / next_state = tableau de 4 elements
# action : 0 ou 1
# reward : 1 si la baton n'est pas tombé
# done : True ou False


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
        if sum(len(item) > 0 for item in self.memory) < self.batch_size:
            # pas assez d'experiences pour construire le batch
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


class ExperienceReplayAgent:
    """
    Agent qui choisit des actions de manière aléatoire
    """
    def __init__(self, action_space, batch_size):
        """
        Initialisation générale
        @param action_space: espace d'actions (0 ou 1)
        @param batch_size: la taille du batch généré via la mémoire
        """
        self.action_space = action_space
        self.memory = Memory(100, 20)  # bien penser a initialiser la memoire pour ne pas avoir d'index out of range
        self.position = 0
        self.batch_size = batch_size

    def act(self):
        """
        Choisit une action aléatoirement parmi l'espace d'actions
        """
        return self.action_space.sample()

    # 2.2 - question 3
    def remember(self, state, action, reward, next_state, done):
        """
        Ajoute une interaction à la mémoire de l'agent
        @param state: état courant
        @param action: action effectuée
        @param reward: récompense reçue de l'environnement
        @param next_state: état dans lequel on arrive
        @param done: pour arrêter l'agent quand il a fini
        """
        # on ajoute l'experience et on incremente la position dans la memoire
        self.memory.add(state, action, reward, next_state, done)

    # 2.2 - question 4
    def creer_batch(self):
        """
        Cree un batch de taille self.batch_size sur la base de la mémoire
        @return le batch
        """
        return self.memory.sample()


# 2.2 - question 4
if __name__ == '__main__':
    env = gym.make("CartPole-v1")  # creation de l'environnement
    agent = ExperienceReplayAgent(env.action_space, 20)  # creation de l'agent (batch_size=20)
    nb_episodes = 100
    done = False  # pour savoir quand on s'arrete (le baton est tombé ou il est sorti de l'environnement)

    for i in range(nb_episodes):
        state = env.reset()
        while True:
            env.render()
            action = agent.act()
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
    batch = agent.creer_batch()  # creation du batch à partir de la memoire de l'agent
    print(batch)
    env.close()
