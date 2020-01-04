
import numpy as np
import time
import gym
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import sys

import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
import os

import numpy as np
import random
class ModelWrapper:
    def __init__(self, model, fit_batch_size=32):
        self._model = model
        self._fit_batch_size = fit_batch_size

    @property
    def action_size(self):
        return self._model.layers[-1].output_shape[1]

    def fit(self, states, predictions, sample_weight=None):
        return self._model.fit(states, predictions, epochs=1, verbose=0, batch_size=self._fit_batch_size,
                               sample_weight=sample_weight)

    def get_weights(self):
        return self._model.get_weights()

    def predict(self, state):
        return self._model.predict(state)

    def set_weights(self, weights):
        self._model.set_weights(weights)

class EnvironmentWrapper:
    def __init__(self, env, n_bootstrap_steps=10000, verbose=1):
        self._env = env
        self._n_samples = 0
        self._mean = None
        self._std = None
        self._verbose = verbose

        if n_bootstrap_steps is not None:
            self._bootstrap(n_bootstrap_steps)

    def _bootstrap(self, n_bootstrap_steps):
        self._mean = None
        self._std = None
        steps = 0

        if self._verbose > 0:
            print('Bootstrapping environment stats over {} random time steps...'.format(n_bootstrap_steps))

        while steps < n_bootstrap_steps:
            done = False
            _ = self.reset()

            while not done:
                steps += 1
                action = random.randrange(self._env.action_space.n)
                _, _, done, _ = self.step(action)

        if self._verbose > 0:
            print('Bootstrapping complete; mean {}, std {}'.format(self._mean, self._std))

    def _update_env_stats(self, sample):
        # Incremental mean/standard deviation
        self._n_samples += 1

        if self._mean is None:
            self._std = np.repeat(1.0, len(sample))
            self._mean = sample
        else:
            self._std = (self._n_samples - 2) / (self._n_samples - 1) * self._std + \
                        (1 / self._n_samples) * np.square(sample - self._mean)
            self._mean += (sample - self._mean) / self._n_samples

    def _standardize(self, state):
        if self._mean is None or self._std is None:
            return state

        return (state - self._mean) / self._std

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def render(self):
        self._env.render()

    def reset(self):
        state = self._env.reset()
        self._update_env_stats(state)
        return self._standardize(state)

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        self._update_env_stats(state)
        return self._standardize(state), reward, done, info
def _smooth_returns(returns, window=10):
    output = [np.nan] * window

    for i in range(window, len(returns)):
        output.append(np.mean(returns[i-window:i]))

    return output


def _plot_series(series, color, label, smooth_window=10):
    series = np.array(series)

    if series.ndim == 1:
        plt.plot(series, color=color, linewidth=0.5)
        plt.plot(_smooth_returns(series, window=smooth_window), color=color, label=label, linewidth=2)
    else:
        mean = series.mean(axis=0)
        plt.plot(mean, color=color, linewidth=1, label=label)
        plt.fill_between(range(series.shape[1]),
                         mean + series.std(axis=0), mean - series.std(axis=0),
                         color=color, alpha=0.2)


def random(env, n_episodes=1000):
    returns = []

    for _ in range(n_episodes):
        _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = np.random.randint(low=0, high=env.action_space.n)
            _, reward, done, _ = env.step(action)
            total_reward += reward

        returns.append(total_reward)

    return returns


def report(returns, render=True, title=None, legend_loc='upper right', smooth_window=10, file=None):
    for i in range(len(returns)):
        series, color, label = returns[i]

        if i == 0:
            print('Experiment stats for {}:'.format(label))
            print('  Mean reward: {}'.format(np.mean(series)))
            print('  Median reward: {}'.format(np.median(series)))
            print('  Std reward: {}'.format(np.std(series)))
            print('  Max reward: {}'.format(np.max(series)))
            print('  Min reward: {}'.format(np.min(series)))

        if not render:
            continue

        _plot_series(series, color=color, label=label, smooth_window=smooth_window)

    if not render:
        return

    if title is not None:
        plt.title(title)

    plt.legend(loc=legend_loc)
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    if file is not None:
        if not os.path.exists('logs'):
            os.makedirs('logs')

        plt.savefig(os.path.join('logs', file))

    plt.show()
class ExponentialSchedule:
    def __init__(self, start, end, step):
        self._value = start
        self._end = end
        self._step = step

    @property
    def value(self):
        return self._value

    def step(self):
        # Simple exponential multiplication step on epsilon (until the end value is reached)
        if self._step < 1:
            self._value = max(self._value * self._step, self._end)
        else:
            self._value = min(self._value * self._step, self._end)


class LinearSchedule:
    def __init__(self, start, end, step):
        self._value = start
        self._end = end
        self._step = step

    @property
    def value(self):
        return self._value

    def step(self):
        # Simple linear change (until end value is met)
        if self._step < 0:
            self._value = max(self._value - self._step, self._end)
        else:
            self._value = min(self._value + self._step, self._end)


class EpsilonGreedyExploration:
    def __init__(self, decay_sched):
        self._schedule = decay_sched

    @property
    def epsilon(self):
        return self._schedule.value

    def act(self, model, state):
        if np.random.rand() <= self._schedule.value:
            return random.randrange(model.action_size)

        # predict() returns a matrix tensor (even for a single state prediction),
        # but the action values are always a vector, so grab the first (and only) row
        return np.argmax(model.predict(state)[0])

    def step(self):
        self._schedule.step()


from collections import deque
import numpy as np
import random


class ExperienceReplay:
    def __init__(self, maxlen, sample_batch_size, min_size_to_sample):
        self._states = deque(maxlen=maxlen)
        self._actions = deque(maxlen=maxlen)
        self._rewards = deque(maxlen=maxlen)
        self._next_states = deque(maxlen=maxlen)
        self._dones = deque(maxlen=maxlen)
        self._sample_batch_size = sample_batch_size
        self._min_size_to_sample = min_size_to_sample

    def __len__(self):
        return len(self._states)

    @property
    def supports_prioritization(self):
        return False

    def add(self, state, action, reward, next_state, done):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._next_states.append(next_state)
        self._dones.append(done)

    def bootstrap(self, env):
        print('Bootstrapping experience samples...')

        while not self.can_sample():
            state = env.reset()
            done = False

            while not done:
                action = np.random.randint(low=0, high=env.action_space.n)
                next_state, reward, done, _ = env.step(action)
                self.add(state, action, reward, next_state, done)

    def can_sample(self):
        return len(self) >= self._min_size_to_sample

    def sample(self):
        mem_size = len(self)
        indices = random.sample(range(mem_size), min(mem_size, self._sample_batch_size))
        states = np.array([self._states[idx] for idx in indices])
        actions = np.array([self._actions[idx] for idx in indices])
        rewards = np.array([self._rewards[idx] for idx in indices])
        next_states = np.array([self._next_states[idx] for idx in indices])
        dones = np.array([self._dones[idx] for idx in indices])
        return states, actions, rewards, next_states, dones, None, indices

    def step(self):
        # No-op
        pass

    def update_priority(self, idx, priority):
        # No-op
        pass


class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, maxlen, sample_batch_size, min_size_to_sample, initial_max_priority=1.0, alpha_sched=None, beta_sched=None):
        super(PrioritizedExperienceReplay, self).__init__(maxlen, sample_batch_size, min_size_to_sample)
        self._priorities = deque(maxlen=maxlen)
        self._alpha_sched = alpha_sched
        self._beta_sched = beta_sched
        self._max_priority = abs(initial_max_priority)

    @property
    def supports_prioritization(self):
        return True

    def add(self, state, action, reward, next_state, done):
        alpha = 1.0 if self._alpha_sched is None else self._alpha_sched.value
        priority = self._max_priority ** alpha

        if self.__len__() < self._states.maxlen:
            # Just append to the end
            self._states.append(state)
            self._actions.append(action)
            self._rewards.append(reward)
            self._next_states.append(next_state)
            self._dones.append(done)
            self._priorities.append(priority)
        else:
            # Replace the smallest existing priority
            min_idx = np.argmin(self._priorities)
            self._states[min_idx] = state
            self._actions[min_idx] = action
            self._rewards[min_idx] = reward
            self._next_states[min_idx] = next_state
            self._dones[min_idx] = done
            self._priorities[min_idx] = priority

    def sample(self):
        dist = np.array(list(self._priorities))
        norm_dist = dist / dist.sum()  # Normalize distribution
        num_samples = self.__len__()

        indices = np.random.choice(range(num_samples), self._sample_batch_size, p=norm_dist)
        states = np.array([self._states[idx] for idx in indices])
        actions = np.array([self._actions[idx] for idx in indices])
        rewards = np.array([self._rewards[idx] for idx in indices])
        next_states = np.array([self._next_states[idx] for idx in indices])
        dones = np.array([self._dones[idx] for idx in indices])

        # Basically an adaption of OpenAI's baselines PER, but I admit I can't understand why this works,
        # as it seems to overweight the SMALLER priorities, rather than the larger priorities.
        # TODO: If someone sees this comment, please help me understand. :)
        beta = 1.0 if self._beta_sched is None else self._beta_sched.value
        dist_max = dist.max()
        dist_min = dist.min() / dist_max
        max_weight = (num_samples * dist_min) ** (-beta)

        samples = dist[indices] / dist_max
        importances = (num_samples * samples) ** (-beta)
        importances /= max_weight

        return states, actions, rewards, next_states, dones, importances, indices

    def step(self):
        if self._alpha_sched is not None:
            self._alpha_sched.step()

        if self._beta_sched is not None:
            self._beta_sched.step()

    def update_priority(self, idx, priority):
        alpha = 1.0 if self._alpha_sched is None else self._alpha_sched.value
        priority = abs(priority) ** alpha
        self._priorities[idx] = priority
        self._max_priority = max(self._max_priority, priority)


import numpy as np


class FixedQTarget:
    def __init__(self, target_model, target_update_step, use_soft_targets=False, use_double_q=False):
        self._target_model = target_model
        self._target_update_step = target_update_step
        self._use_soft_targets = use_soft_targets
        self._tau = 1.0 / self._target_update_step
        self._n_steps = 0
        self._use_double_q = use_double_q

    @property
    def use_double_q(self):
        return self._use_double_q

    def predict(self, states):
        return self._target_model.predict(states)

    def reset(self, policy_model):
        self._target_model.set_weights(policy_model.get_weights())
        self._n_steps = 0

    def step(self, policy_model):
        if self._use_soft_targets:
            # Soft update fixed-Q targets
            weights_model = policy_model.get_weights()
            weights_target = self._target_model.get_weights()
            new_weights = []

            for i in range(len(weights_model)):
                new_weights.append(self._tau * weights_model[i] + (1. - self._tau) * weights_target[i])

            self._target_model.set_weights(new_weights)
        else:
            if self._n_steps % self._target_update_step == 0:
                self._target_model.set_weights(policy_model.get_weights())

    def swap_models(self, policy_model):
        # To reduce/eliminate maximization bias, the target and policy networks should be randomly swapped
        # See Hasselt 2010: https://papers.nips.cc/paper/3964-double-q-learning.pdf
        if self._use_double_q and np.random.random() < 0.5:
            swapped_model = self._target_model
            self._target_model = policy_model
            return swapped_model
        else:
            return policy_model


import keras.backend as K
import tensorflow as tf


def huber_loss(target, prediction, clip_delta=1.0):
    # Implementation from: https://stackoverflow.com/a/48791563/1207773
    error = target - prediction
    cond = tf.keras.backend.abs(error) < clip_delta
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)


def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    # Implementation from: https://stackoverflow.com/a/48791563/1207773
    return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))


def pseudo_huber_loss(target, prediction, delta=1.0):
    # Pseudo-Huber loss from: https://en.wikipedia.org/wiki/Huber_loss
    error = prediction - target
    return K.mean(K.sqrt(1 + K.square(error / delta)) - 1, axis=-1)


class DQNAgent:
    def __init__(self, env, model, gamma, exploration, experience=None, fixed_q_target=None, n_steps=1):
        self._env = env
        self._model = model
        self._gamma = gamma
        self._exploration = exploration
        self._experience = experience
        self._fixed_q_target = fixed_q_target
        self._n_steps = n_steps

        if self._fixed_q_target is not None:
            self._fixed_q_target.reset(self._model)

    @property
    def exploration(self):
        return self._exploration

    def _get_predictions(self, samples):
        states, actions, rewards, next_states, dones, sample_weights, sample_indices = samples
        predictions = np.zeros((len(states), self._model.action_size))

        action_returns = self._model.predict(states)
        next_action_returns = self._get_next_action_returns(next_states)
        sampled_returns = []

        for idx in range(len(states)):
            action, reward, done, action_return = actions[idx], rewards[idx], dones[idx], action_returns[idx]
            policy_action = self._select_policy_action(next_states, next_action_returns, idx)
            discounted_return = self._gamma * next_action_returns[idx][policy_action] * (not done)
            action_return[action] = reward + discounted_return
            predictions[idx] = action_return
            sampled_returns.append(action_return)

            if self._experience is not None and self._experience.supports_prioritization:
                importance, sample_idx = sample_weights[idx], sample_indices[idx]
                td_error = (action_return - self._model.predict(np.array([states[idx]])))[0][action]
                self._experience.update_priority(sample_idx, td_error)

        return predictions, sample_weights, sampled_returns

    def _get_next_action_returns(self, next_states):
        if self._fixed_q_target is not None:
            # Fixed-Q targets use next action returns from the target policy (off-policy)
            return self._fixed_q_target.predict(next_states)
        else:
            # Get the next action returns from the on-policy model
            return self._model.predict(next_states)

    def _sample_experience(self, state, action, reward, next_state, done):
        if self._experience is not None:
            # if self._experience.supports_prioritization:
            #     # Schaul 2015 says that incoming samples have no known error, but this seems incorrect; it seems
            #     # we can estimate the sample's TD error upon arrival.
            #     # Jaromir Janisch makes the same conclusion; see: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
            #     action_return = reward + self._get_next_action_returns(np.array([next_state]))
            #     td_error = (action_return - self._model.predict(np.array([state])))[0][action]
            # else:
            #     td_error = None

            self._experience.add(state, action, reward, next_state, done)
            return self._experience.sample()
        else:
            # This is a "vanilla" DQN
            return np.array([state]), np.array([action]), np.array([reward]), np.array([next_state]), np.array([done]),\
                   None, None

    def _select_policy_action(self, next_states, next_action_returns, sample_idx):
        if self._fixed_q_target is not None and self._fixed_q_target.use_double_q:
            # Double-Q selects the greedy action of the on-policy model (but evaluates it off-policy)
            return np.argmax(self._model.predict(next_states)[sample_idx])
        else:
            # Select the greedy action from the action returns given
            return np.argmax(next_action_returns[sample_idx])

    def _do_n_steps(self, action):
        # TODO: Complete redo this
        next_state = None
        reward = 0
        done = False
        step_action = action

        for i in range(self._n_steps):
            step_next_state, step_reward, step_done, _ = self._env.step(step_action)
            reward += step_reward * (self._gamma ** i)

            if next_state is None:
                # Just store the "first" next state when using multiple N-steps
                next_state = step_next_state

            if done is None:
                # Just store the "first" done when using multiple N-steps
                done = step_done

            if step_done:
                break

            # Get the greedy action using the on-policy model
            step_action = np.argmax(self._model.predict(np.array([next_state]))[0])

        return next_state, reward, done, dict()

    def train(self, render=False, debug_func=None):
        state = self._env.reset()
        total_reward = 0
        done = False
        n_steps = 0
        start_time = time.time()
        step_rewards = []
        losses = []

        while not done:
            if render:
                self._env.render()

            action = self._exploration.act(self._model, np.array([state]))
            # next_state, reward, done, _ = self._do_n_steps(action)
            next_state, reward, done, _ = self._env.step(action)
            samples = self._sample_experience(state, action, reward, next_state, done)

            if self._fixed_q_target is not None:
                self._fixed_q_target.step(self._model)

            states = samples[0]
            predictions, sample_weights, sampled_returns = self._get_predictions(samples)
            history = self._model.fit(states, predictions, sample_weight=sample_weights)
            losses.extend(history.history['loss'])

            # The Hasselt 2010 algorithm calls for randomly swapping models each update step.
            # This isn't seen much in DQN implementations, but seems more theoretically sound.
            if self._fixed_q_target is not None:
                self._model = self._fixed_q_target.swap_models(self._model)

            state = next_state
            total_reward += reward
            n_steps += 1
            step_rewards.extend(sampled_returns)

        if self._experience is not None:
            self._experience.step()

        self._exploration.step()

        # Allow the chance to examine the model for debugging
        if debug_func is not None:
            debug_func(self._model)

        elapsed_time = time.time() - start_time
        return total_reward, n_steps, elapsed_time, np.mean(step_rewards), np.mean(losses)

    def test(self, render=False, verbose=0):
        state = self._env.reset()
        total_reward = 0
        done = False
        n_steps = 0
        step_rewards = []

        while not done:
            if render:
                self._env.render()

            # Act greedily during testing
            action = np.argmax(self._model.predict(np.array([state]))[0])
            next_state, reward, done, _ = self._env.step(action)

            if verbose > 0:
                print('Step {}: reward {}, state {}, action {}'.format(n_steps, reward, state, action))

            state = next_state
            total_reward += reward
            n_steps += 1
            step_rewards.append(reward)

        return step_rewards, n_steps






def build_network(env, verbose=True):
    model = Sequential()
    model.add(Dense(24, input_dim=env.observation_space.shape[0], activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    if verbose:
        model.summary()

    return ModelWrapper(model)


def train_dqn(agent, n_episodes=None, debug=False):
    # Experiment described by: https://github.com/openai/gym/wiki/CartPole-v0
    # CartPole-v1 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.
    # This environment corresponds to the version of the cart-pole problem described by
    # Barto, Sutton, and Anderson [Barto83].
    exp_returns = []
    training_complete = False
    e = 0
    action_vals = []

    def debug_func(model):
        # Just an arbitrary first state/action pair from a new episode of a fully trained model
        state = np.array([[0.3604471, 0.21131558, 5.13830467, 0.07171951]])
        action = 0
        x = model.predict(state)[0][action]
        action_vals.append(x)

    # Arbitrary maximum at 2000 episodes, in case of divergent training
    while not training_complete and e < 2000:
        e += 1
        total_reward, n_steps, elapsed_time, _, _ = agent.train(debug_func=debug_func if debug else None)
        exp_returns.append(total_reward)

        print('Episode {} took {} steps and got {} reward in {} seconds; epsilon now {}'.format(
            e, n_steps, total_reward, elapsed_time, agent.exploration.epsilon))

        if n_episodes is not None:
            training_complete = e == n_episodes
        else:
            # CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.
            training_complete = np.mean(exp_returns[-100:]) >= 195

    print('Training complete after {} episodes'.format(e))

    if debug:
        plt.plot(exp_returns, color='b', label='Rewards')
        plt.plot(action_vals, color='r', label='Q-value')
        plt.legend(loc='upper left')
        plt.show()

    if n_episodes is None:
        step_rewards, n_steps = agent.test(render=True, verbose=1)
        print('Testing: {} total reward over {} steps'.format(np.sum(step_rewards), n_steps))

    return exp_returns


def data_exploration(env, n_episodes):
    # Random exploration to establish a baseline
    exp_returns = data.random(env, n_episodes=n_episodes)
    return exp_returns


def basic_dqn(env, n_episodes):
    # Basic DQN with e-greedy exploration
    model = build_network(env)
    decay_sched = ExponentialSchedule(start=1.0, end=0.01, step=0.99)
    exploration = EpsilonGreedyExploration(decay_sched=decay_sched)
    agent = DQNAgent(env, model, gamma=0.99, exploration=exploration)

    # Perform the training
    return train_dqn(agent, n_episodes, debug=n_episodes is None)


def dqn_with_experience(env, n_episodes):
    # DQN with e-greedy exploration and experience replay
    model = build_network(env)
    experience = ExperienceReplay(maxlen=2000, sample_batch_size=32, min_size_to_sample=100)
    decay_sched = ExponentialSchedule(start=1.0, end=0.01, step=0.99)
    exploration = EpsilonGreedyExploration(decay_sched=decay_sched)
    agent = DQNAgent(env, model, gamma=0.99, exploration=exploration, experience=experience)

    # Pre-load samples in experience replay.
    # This can also be done implicitly during regular training episodes,
    # but the early training may overfit to early samples.
    experience.bootstrap(env)

    # Perform the training
    return train_dqn(agent, n_episodes, debug=n_episodes is None)


def dqn_with_fixed_targets(env, n_episodes=None):
    # DQN with e-greedy exploration, experience replay, and fixed-Q targets
    model = build_network(env)
    target_model = build_network(env)
    experience = ExperienceReplay(maxlen=2000, sample_batch_size=32, min_size_to_sample=100)
    decay_sched = ExponentialSchedule(start=1.0, end=0.01, step=0.99)
    exploration = EpsilonGreedyExploration(decay_sched=decay_sched)
    fixed_target = FixedQTarget(target_model, target_update_step=500, use_soft_targets=True)
    agent = DQNAgent(env, model, gamma=0.99, exploration=exploration, experience=experience, fixed_q_target=fixed_target)

    # Pre-load samples in experience replay.
    # This can also be done implicitly during regular training episodes,
    # but the early training may overfit to early samples.
    experience.bootstrap(env)

    # Perform the training
    return train_dqn(agent, n_episodes, debug=n_episodes is None)


def run_single_trials():
    env = EnvironmentWrapper(gym.make('CartPole-v1'))
    n_episodes = 500

    baseline_returns = data_exploration(env, n_episodes)
    data.report([(baseline_returns, 'b', 'Baseline')], title='Random Walk', file='cartpole_single_random_walk.png')

    basic_dqn_returns = basic_dqn(env, n_episodes)
    data.report([(basic_dqn_returns, 'b', 'Basic DQN'),
                 (baseline_returns, 'r', 'Baseline')], title='Vanilla DQN', file='cartpole_single_basic_dqn.png')

    dqn_w_exp_returns = dqn_with_experience(env, n_episodes)
    data.report([(dqn_w_exp_returns, 'b', 'DQN w/ ER'),
                 (baseline_returns, 'r', 'Baseline')], title='Experience Replay', file='cartpole_single_er_dqn.png')

    dqn_w_fixed_targets_returns = dqn_with_fixed_targets(env, n_episodes)
    data.report([(dqn_w_fixed_targets_returns, 'b', 'DQN w/ Fixed-Q'),
                 (baseline_returns, 'r', 'Baseline')], title='Fixed-Q Targets', file='cartpole_single_fixedq_dqn.png')

    # Plot all the variations
    data.report([(basic_dqn_returns, 'b', 'Basic DQN'),
                 (dqn_w_exp_returns, 'g', 'DQN w/ ER'),
                 (dqn_w_fixed_targets_returns, 'm', 'DQN w/ Fixed-Q'),
                 (baseline_returns, 'r', 'Baseline')], title='All DQN Variants', file='cartpole_single_all_dqn.png')


def run_multiple_trials():
    env = EnvironmentWrapper(gym.make('CartPole-v1'))
    n_episodes = 500
    n_trials = 10

    baseline_returns = []

    for i in range(n_trials):
        baseline_returns.append(data_exploration(env, n_episodes))

    data.report([(baseline_returns, 'b', 'Baseline')], title='Random Walk', file='cartpole_multi_random_walk.png')

    basic_dqn_returns = []

    for i in range(n_trials):
        basic_dqn_returns.append(basic_dqn(env, n_episodes))

    data.report([(basic_dqn_returns, 'b', 'Basic DQN'),
                 (baseline_returns, 'r', 'Baseline')], title='Vanilla DQN', file='cartpole_multi_basic_dqn.png')

    dqn_w_exp_returns = []

    for i in range(n_trials):
        dqn_w_exp_returns.append(dqn_with_experience(env, n_episodes))

    data.report([(dqn_w_exp_returns, 'b', 'DQN w/ ER'),
                 (baseline_returns, 'r', 'Baseline')], title='Experience Replay', file='cartpole_multi_er_dqn.png')

    dqn_w_fixed_targets_returns = []

    for i in range(n_trials):
        dqn_w_fixed_targets_returns.append(dqn_with_fixed_targets(env, n_episodes))

    data.report([(dqn_w_fixed_targets_returns, 'b', 'DQN w/ Fixed-Q'),
                 (baseline_returns, 'r', 'Baseline')], title='Fixed-Q Targets', file='cartpole_multi_fixedq_dqn.png')

    # Plot all the variations
    data.report([(basic_dqn_returns, 'b', 'Basic DQN'),
                 (dqn_w_exp_returns, 'g', 'DQN w/ ER'),
                 (dqn_w_fixed_targets_returns, 'm', 'DQN w/ Fixed-Q'),
                 (baseline_returns, 'r', 'Baseline')], title='All DQN Variants', file='cartpole_multi_all_dqn.png')


def solve():
    env = EnvironmentWrapper(gym.make('CartPole-v1'))
    n_episodes = []

    for i in range(10):
        returns = dqn_with_fixed_targets(env, n_episodes=None)
        n_episodes.append(len(returns))

        if i == 0:
            baseline_returns = data_exploration(env, n_episodes=len(n_episodes))
            data.report([(returns, 'b', 'Solution'),
                         (baseline_returns, 'r', 'Baseline')], title='Solution', file='cartpole_solve_fixedq_dqn.png')

    n_episodes = np.array(n_episodes)
    print('CartPole solved!')
    print('  Median: {} episodes'.format(np.median(n_episodes)))
    print('  Mean:   {} episodes'.format(np.mean(n_episodes)))
    print('  Std:    {} episodes'.format(np.std(n_episodes)))
    print('  Min:    {} episodes'.format(np.min(n_episodes)))
    print('  Max:    {} episodes'.format(np.max(n_episodes)))
    print('  % diverged: {}'.format(len(n_episodes[n_episodes >= 2000]) / float(len(n_episodes))))

    ''' Sample output:
CartPole solved!
  Median: 309.0 episodes
  Mean:   422.3 episodes
  Std:    241.73624055983004 episodes
  Min:    239 episodes
  Max:    1075 episodes
  % diverged: 0.0
    '''


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
    else:
        arg = 'single'

    if arg == 'multiple':
        run_multiple_trials()
    elif arg == 'solve':
        solve()
    else:
        run_single_trials()


if __name__ == "__main__":
    main()