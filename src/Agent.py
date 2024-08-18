import numpy as np
import pickle
from sklearn import neural_network


class ExperienceReplay(object):
    def __init__(self, max_memory=5000, discount=0.9):
        self.memory = []
        self.max_memory = max_memory
        self.discount = discount

    def remember(self, experience):
        self.memory.append(experience)

    # The sampling probability is unequal, and experiences that are new to the experience playback array have a high probability of being sampled.
    def get_batch(self, batch_size=10):

        if len(self.memory) > self.max_memory:
            del self.memory[:len(self.memory) - self.max_memory]

        if batch_size < len(self.memory):
            timerank = range(1, len(self.memory) + 1)
            p = timerank / np.sum(timerank, dtype=float)
            batch_idx = np.random.choice(range(len(self.memory)), replace=False, size=batch_size, p=p)
            batch = [self.memory[idx] for idx in batch_idx]
        else:
            batch = self.memory

        return batch


# This is the base agent, and all subsequent agents are subclasses of it.
class BaseAgent(object):
    def __init__(self, histlen):
        self.single_testcases = True
        self.train_mode = True
        self.histlen = histlen

    def get_action(self, s):
        return 0

    def get_all_actions(self, states):
        """ Returns list of actions for all states """
        return [self.get_action(s) for s in states]

    def reward(self, reward):
        pass

    def save(self, filename):
        """ Stores agent as pickled file """
        pickle.dump(self, open(filename + '.p', 'wb'), 2)

    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename + '.p', 'rb'))


class NetworkAgent(BaseAgent):

    def __init__(self, action_size, hidden_size, histlen, name):
        super(NetworkAgent, self).__init__(histlen=histlen)
        self.name = name
        self.experience_length = 10000
        self.experience_batch_size = 1000  # The amount of experience taken out of the experience array each time
        self.experience = ExperienceReplay(max_memory=self.experience_length)
        self.episode_history = []  # Executive history
        self.iteration_counter = 0
        self.action_size = action_size

        if isinstance(hidden_size, tuple):
            self.hidden_size = hidden_size
        else:
            self.hidden_size = (hidden_size,)

        self.model = None
        self.model_fit = False
        self.init_model(True)

    def init_model(self, warm_start=True):
        if self.action_size == 1:

            self.model = neural_network.MLPClassifier(hidden_layer_sizes=self.hidden_size, activation='relu',
                                                      warm_start=warm_start, solver='adam', max_iter=750)
        else:
            self.model = neural_network.MLPRegressor(hidden_layer_sizes=self.hidden_size, activation='relu',
                                                     warm_start=warm_start, solver='adam', max_iter=750)
        self.model_fit = False

    def get_action(self, s):
        if self.model_fit:
            if self.action_size == 1:
                a = self.model.predict_proba(np.array(s).reshape(1, -1))[0][1]
            else:
                a = self.model.predict(np.array(s).reshape(1, -1))[0]
        else:
            a = np.random.random()

        if self.train_mode:
            self.episode_history.append((s, a))

        return a

    def reward(self, rewards, experceInPoll=True):

        if not self.train_mode:
            return

        try:
            # Reward is a string or a number
            x = float(rewards)
            rewards = [x] * len(self.episode_history)
        except:
            if len(rewards) < len(self.episode_history):
                raise Exception('Too few rewards')

        self.iteration_counter += 1

        if experceInPoll or self.iteration_counter == 1:
            for ((state, action), reward) in zip(self.episode_history, rewards):
                self.experience.remember((state, reward))  # Experience deposited into the experience pool

        self.episode_history = []

        # timing training
        if experceInPoll == False or self.iteration_counter == 1 or self.iteration_counter % 5 == 0:
            self.learn_from_experience()

    def learn_from_experience(self):
        experiences = self.experience.get_batch(self.experience_batch_size)
        x, y = zip(*experiences)

        if self.model_fit:
            try:
                # Update the model with a single iteration over the given data.
                self.model.partial_fit(x, y)
            except ValueError:
                self.init_model(warm_start=False)

                # Fit the model to data matrix X and target(s) y.
                # y: The target values (class labels in classification, real numbers in regression).
                self.model.fit(x, y)

                self.model_fit = True
        else:
            self.model.fit(x, y)  # Call fit once to learn classes
            self.model_fit = True
