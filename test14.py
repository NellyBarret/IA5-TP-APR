import numpy as np


class rl_memory(object):
    """Data storage and batch retrieval class for DQN"""

    def __init__(self, capacity, batch_size, seed):
        self.capacity = capacity
        self.states = np.zeros((self.capacity, 4))
        self.actions = np.zeros(self.capacity, dtype=np.int)
        self.rewards = np.zeros(self.capacity)
        self.next_states = np.zeros((self.capacity, 4))
        self.current = 0

    def add(self, state, action, reward, next_state):
        self.states[self.current] = state
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.next_states[self.current] = next_state
        self.current = (self.current + 1) % self.capacity

    def get_batch(self, batch_size):
        indexes = np.random.choice(min(self.capacity, self.current), batch_size, replace=False)
        return self.states[indexes], self.actions[indexes], self.rewards[indexes], self.next_states[indexes]


import tensorflow as tf
import numpy as np


class rl_model(object):
    """Neural Network Model Class for CartPole DQN"""

    def __init__(self, learning_rate, discount_factor, seed):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.session = tf.Session()
        self.set_model()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _add_layer(self, in_dim, out_dim, input):
        """Add a layer to the neural network"""
        w = tf.Variable(initial_value=tf.random_normal(shape=(in_dim, out_dim), dtype=tf.float32))
        b = tf.Variable(initial_value=tf.zeros(shape=(1, out_dim), dtype=tf.float32))

        return tf.matmul(input, w) + b

    def save(self, number):
        """ Save model in ./tmp/ directory with number specified in filename"""
        return self.saver.save(self.session, "./tmp/model_%3.f.ckpt" % (number))

    def load(self, name):
        """ Load model from ./tmp/ directory with number specified filename"""
        self.saver.restore(self.session, "./tmp/model_%3.f.ckpt" % (name))

    def set_model(self):
        self._x = tf.placeholder(tf.float32, shape=(None, 4), name="X")
        self._y = tf.placeholder(tf.float32, shape=(None, 2), name="Y")

        z1 = self._add_layer(4, 8, self._x)
        a1 = tf.nn.relu(z1)
        z2 = self._add_layer(8, 8, a1)
        a2 = tf.nn.relu(z2)
        self.y_ = self._add_layer(8, 2, a2)

        self.loss = tf.losses.mean_squared_error(self._y, self.y_)
        self.step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def predict(self, states):
        """ Use model to make a prediction on data"""
        return self.session.run(self.y_, feed_dict={self._x: states})

    def _preprocess(self, data):
        """ Helper function: takes data from environment and converts to x,y pairs for training"""
        y_next = np.max(self.predict(data[3]), axis=1)
        y_target = self.predict(data[0])
        for i in range(len(data[0])):
            y_target[i, data[1][i]] = data[2][i] + self.discount_factor * y_next[i]
        return data[0], y_target

    def train(self, data):
        """ Runs a training iteration on data and returns loss"""
        x, y = self._preprocess(data)
        loss, _ = self.session.run([self.loss, self.step],
                                   feed_dict={self._x: x, self._y: y})
        return loss



import gym

env = gym.make('CartPole-v1')
env.reset()

for _ in range(2000):
    env.render()
    s, r, done, info = env.step(env.action_space.sample())
    if _ % 300 == 0:
        # restarts environment occasionally
        # Note: the game is usually terminated long before restart
        # as the action game mecanism return done when the pole tilts
        # more than 15 degrees from vertical.
        env.reset()

env.close()
print('done')

import numpy as np
import gym

# *** Change this number to the number of a different model to use that model ***
MODEL_NUMBER = 441

NUM_TRAIN_GAMES = 25000
NUM_TEST_GAMES = 100
NUM_TEST_VISUAL_GAMES = 10
MAX_GAME_STEPS = 500
LOSS_PENALTY = -100
RANDOM_SEED = 0
MEMORY_CAPACITY = 1000000
BATCH_SIZE = 256
GYM_ENVIRONMENT = 'CartPole-v1'
LEARNING_RATE = 1e-2
DISCOUNT_FACTOR = 0.8
START_EPSILON = 1.0
MIN_EPSILON = 0.5
EPSILON_DECAY = 0.999

model = rl_model(LEARNING_RATE, DISCOUNT_FACTOR, RANDOM_SEED)
env = gym.make(GYM_ENVIRONMENT)


def run_test(render=False, num_games=NUM_TEST_GAMES):
    times = []
    sides = [0, 0]
    t = 0
    for i in range(num_games):
        state = env.reset()
        while (True):
            if render:
                env.render()
            action = np.argmax(model.predict([state]), axis=1)[0]
            next_state, reward, done, info = env.step(action)
            sides[action] += 1

            state = next_state
            t += 1
            if done:
                break
        times.append(t)
        state = env.reset()
        t = 0
    #     print(sides)
    return times


model.load(MODEL_NUMBER)

print(run_test(render=True, num_games=NUM_TEST_VISUAL_GAMES))

env.close()
print("Done")

""" Main Class to train model """

import numpy as np
import gym

NUM_TRAIN_GAMES = 5000
NUM_TEST_GAMES = 100
NUM_TEST_VISUAL_GAMES = 3
MAX_GAME_STEPS = 500
NUM_RANDOM_GAMES = 50
TEST_FREQUENCY = 200
LOSS_PENALTY = -100
RANDOM_SEED = 0
MEMORY_CAPACITY = 10000000
BATCH_SIZE = 256
GYM_ENVIRONMENT = 'CartPole-v1'
LEARNING_RATE = 3e-3
DISCOUNT_FACTOR = 0.8
START_EPSILON = 1.0
MIN_EPSILON = 0.5
EPSILON_DECAY = 0.999

model = rl_model(LEARNING_RATE, DISCOUNT_FACTOR, RANDOM_SEED)
memory = rl_memory(MEMORY_CAPACITY, BATCH_SIZE, RANDOM_SEED)
env = gym.make(GYM_ENVIRONMENT)


def run_test(render=False, num_games=NUM_TEST_GAMES):
    times = []
    sides = [0, 0]
    for i in range(num_games):
        state = env.reset()
        for j in range(MAX_GAME_STEPS):
            if render:
                env.render()
            action = np.argmax(model.predict([state]), axis=1)[0]
            next_state, reward, done, info = env.step(action)
            sides[action] += 1

            state = next_state

            if done:
                break
        times.append(j)
        state = env.reset()
    # Uncomment to print number of occurences of each action
    # print(sides)
    return times


def get_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(model.predict([state]), axis=1)[0]


def run_train():
    loss = 0
    prev_avg_time = 0
    epsilon = START_EPSILON
    for i in range(NUM_TRAIN_GAMES):
        state = env.reset()
        for j in range(MAX_GAME_STEPS):
            action = get_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            if done:
                if i > NUM_RANDOM_GAMES and epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY
                reward = LOSS_PENALTY

            memory.add(state, action, reward, next_state)
            state = next_state

            if done:
                break
        if i > NUM_RANDOM_GAMES:
            loss += model.train(memory.get_batch(BATCH_SIZE))
        if i > NUM_RANDOM_GAMES and i % TEST_FREQUENCY == 0:
            times = run_test()
            avg_time = sum(times) / len(times)
            if avg_time > 200 and int(avg_time) > prev_avg_time:
                prev_avg_time = avg_time
                print(model.save(avg_time))
            print('Game:', i)
            print('Loss:', loss / TEST_FREQUENCY)
            print('Time:', avg_time)
            #             print('Epsilon', epsilon)
            print()
            loss = 0


run_train()
# run_test(render=True, num_games=NUM_TEST_VISUAL_GAMES)
print(run_test())
env.close()

print("Done")