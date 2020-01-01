import torch
from torch import nn
import torch.nn.functional as F
import random
import gym
import numpy


class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = [[] for _ in range(self.max_size)]
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """
        Ajoute une transition à la mémoire de l'agent
        """
        self.memory[self.position] = [state, action, reward, next_state, done]
        self.position = (self.position + 1) % self.max_size  # modulo la taille max pour ne pas depasser

    def create_batch(self, batch_size):
        """
        Creer un batch de taille `batch_size` à partir de la mémoire de l'agent
        """
        return random.sample(self.memory, batch_size)


EXPLORATION_RATE = 0
class DQNAgent(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.linear1 = nn.Linear(4, 256)
        self.linear2 = nn.Linear(256, 2)
        self.memory = ReplayMemory(100000)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def select_action(state):
    if numpy.random.rand() < EXPLORATION_RATE:
        # on prend une action aléatoirement parmi les actions possibles
        return random.randrange(self.action_space)
    else:
        with torch.no_grad():
            return policy_network(state).max(1)[1]  # TODO: argmax ?



if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = DQNAgent()
    policy_network = DQNAgent()
    target_network = DQNAgent()