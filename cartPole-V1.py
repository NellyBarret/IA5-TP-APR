import gym
import matplotlib.pyplot as plt


class ReplayAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.memory = []

    def act(self, observation, reward, done):
        return self.action_space.sample()


env = gym.make("CartPole-v1")
observation = env.reset()
list_reward = []
list_iteration = []
somme_reward = 0

agent = ReplayAgent(env.action_space)
episode_count = 100
reward = 0
done = False

for i in range(episode_count):
    ob = env.reset()
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        if done:
            print("episode ", i, ", somme reward : ", somme_reward)
            list_reward.append(somme_reward)
            list_iteration.append(i)

            break


# for i in range(1000):
#     env.render()
#     action = agent.act(ob, reward, done) # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)
#     somme_reward += reward
#
#     if done:
#         print("episode ", i, ", somme reward : ", somme_reward)
#         list_reward.append(somme_reward)
#         list_iteration.append(i)
#         somme_reward = 0
#         observation = env.reset()

fig = plt.figure()
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(111)
ax1.set_ylabel('somme des récompenses')
ax1.set_xlabel('épisode')
ax1.set_title('cartPole - v1')
ax1.plot(list_iteration, list_reward)

plt.show()
env.close()
