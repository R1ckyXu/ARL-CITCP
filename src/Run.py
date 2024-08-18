from datetime import datetime
import numpy as np
import os
import time
import multiprocessing
import pickle
import warnings

warnings.simplefilter("ignore")

from Agent import NetworkAgent, ExperienceReplay
from Env import get_scenario

from Reward import ATCF5

from Visualization import visualize

DEFAULT_NO_SCENARIOS = 1000
DEFAULT_NO_ACTIONS = 100
DEFAULT_HISTORY_LENGTH = 4
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_EPSILON = 0.2
DEFAULT_DUMP_INTERVAL = 100
DEFAULT_VALIDATION_INTERVAL = 100
DEFAULT_PRINT_LOG = False
DEFAULT_PLOT_GRAPHS = False
DEFAULT_NO_HIDDEN_NODES = 12
DEFAULT_PTACTIONSIZE = 7

DEFAULT_TODAY = datetime.today()
print('DEFAULT_TODAY :', DEFAULT_TODAY)

ITERATIONS = 30  # Number of times the experiment is repeated
CI_CYCLES = 1000

USE_LATEX = False
DATA_DIR = 'results'
FIGURE_DIR = 'results_csv'
# PARALLEL = False
PARALLEL = True
PARALLEL_POOL_SIZE = 4

RUN_EXPERIMENT = True
VISUALIZE_RESULTS = True


def preprocess_continuous(state, scenario_metadata, histlen):
    if scenario_metadata['maxExecTime'] > scenario_metadata['minExecTime']:
        time_since = (scenario_metadata['maxExecTime'] - state['LastRun']).total_seconds() / (
                scenario_metadata['maxExecTime'] - scenario_metadata['minExecTime']).total_seconds()
    else:
        time_since = 0

    history = [1 if res else 0 for res in state['LastResults'][0:histlen]]

    if len(history) < histlen:
        history.extend([1] * (histlen - len(history)))

    row = [
        state['Duration'] / scenario_metadata['totalTime'],
        time_since
    ]
    row.extend(history)

    return tuple(row)


# get a priority
def process_scenario_new(agent, sc, preprocess):
    scenario_metadata = sc.get_ta_metadata()

    for row in sc.testcases():
        # Build input vector: preprocess the observation
        # row  is a single testcase  ['Id', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'LastResults']
        x = preprocess(row, scenario_metadata, agent.histlen)

        if len(row["LastResults"]) == 0 or row[
            "intervalCycle"] >= 2:  # Newly emerging test cases have their priority set directly to 1
            action = 1.0
            agent.episode_history.append((x, action))
        else:
            action = agent.get_action(x)

        row['CalcPrio'] = action  # Store prioritization

    # Submit prioritized file for evaluation
    # step the environment and get new measurements
    return sc.submit()


class PrioLearning(object):
    def __init__(self, agent, scenario_provider, file_prefix, reward_function, output_dir, preprocess_function,
                 dump_interval=DEFAULT_DUMP_INTERVAL, validation_interval=DEFAULT_VALIDATION_INTERVAL):
        self.agent = agent
        self.scenario_provider = scenario_provider
        self.reward_function = reward_function
        self.preprocess_function = preprocess_function
        self.replay_memory = ExperienceReplay()
        self.validation_res = []

        self.dump_interval = dump_interval
        self.validation_interval = validation_interval

        self.today = DEFAULT_TODAY

        self.file_prefix = file_prefix
        self.val_file = os.path.join(output_dir, '%s_val' % file_prefix)
        self.stats_file = os.path.join(output_dir, '%s_stats' % file_prefix)
        self.agent_file = os.path.join(output_dir, '%s_agent' % file_prefix)

        # return-based
        self.ut = 0

    def process_scenario(self, sc):

        result = process_scenario_new(self.agent, sc, self.preprocess_function)

        reward = self.reward_function(result, sc)
        sumr = sum(reward)
        if sumr <= self.ut:
            self.agent.reward(reward, experceInPoll=False)
            result = process_scenario_new(self.agent, sc, self.preprocess_function)  # repredict
            reward = self.reward_function(result, sc)
            self.agent.reward(reward)
        else:
            self.agent.reward(reward)

        self.ut = sumr
        return result, reward

    def replay_experience(self, batch_size):
        batch = self.replay_memory.get_batch(batch_size)

        for sc in batch:
            (result, reward, apr, nor) = self.process_scenario(sc)
            print('Replay Experience: %s / %.2f' % (result, np.mean(reward)))

    # Variance of reward values
    def comput2(self, lista):
        ave = np.mean(lista)
        newl = []
        try:
            for i in range(len(lista)):
                newl.append((lista[i] - ave) ** 2)
            return np.mean(newl)
        except:
            return 0

    def train(self):
        stats = {
            'agent': self.agent.name,
            'scenarios': [],
            'rewards': [],
            'rewards_variance': [],
            'durations': [],
            'detected': [],
            'missed': [],
            'ttf': [],
            'napfd': [],
            'recall': [],
            'avg_precision': [],
            'result': [],
            'step': [],
            'env': self.scenario_provider.name,
            'rewardfun': self.reward_function.__name__,
        }
        sum_scenarios = 0

        for (i, sc) in enumerate(self.scenario_provider, start=1):
            start = time.time()

            (result, reward) = self.process_scenario(sc)
            end = time.time()

            sum_scenarios += 1
            duration = end - start

            stats['scenarios'].append(sc.name)
            stats['rewards'].append(np.mean(reward))
            stats['rewards_variance'].append(self.comput2(reward))
            stats['durations'].append(duration)
            stats['detected'].append(result[0])
            stats['missed'].append(result[1])
            stats['ttf'].append(result[2])
            stats['napfd'].append(result[3])
            stats['recall'].append(result[4])
            stats['avg_precision'].append(result[5])
            stats['result'].append(result)
            stats['step'].append(sum_scenarios)

        # end for
        if self.dump_interval > 0:
            # self.agent.save(self.agent_file)
            pickle.dump(stats, open(self.stats_file + '.p', 'wb'))

        return np.mean(stats['napfd'])


def exp_run_industrial_datasets(iteration):
    ags = [
        lambda: (NetworkAgent(histlen=DEFAULT_HISTORY_LENGTH,
                              action_size=1,
                              hidden_size=DEFAULT_NO_HIDDEN_NODES,
                              name="skcla"),
                 preprocess_continuous),

    ]

    datasets = ['apache_drill', 'apache_commons', "apache_parquet", "paintcontrol", 'iofrol',
                'dspace', 'google_auto', 'apache_tajo', 'google_closure', 'google_guava',
                'mybatis', 'rails']

    reward_funs = {

        "ATCF5": ATCF5,

    }

    avg_napfd = []

    for i, get_agent in enumerate(ags):
        for sc in datasets:
            for (reward_name, reward_fun) in reward_funs.items():
                agent, preprocessor = get_agent()
                print('iteration = {}, dataset = {}, reward = {}, agent={}'.format(iteration, sc, reward_name,
                                                                                   agent.name))
                file_appendix = 'rq_%s_%s_%s_%d' % (agent.name, sc, reward_name, iteration)

                scenario = get_scenario(sc)

                rl_learning = PrioLearning(agent=agent,
                                           scenario_provider=scenario,
                                           reward_function=reward_fun,
                                           preprocess_function=preprocessor,
                                           file_prefix=file_appendix,
                                           dump_interval=100,
                                           validation_interval=0,
                                           output_dir=DATA_DIR)

                res = rl_learning.train()
                avg_napfd.append(res)

    return avg_napfd


def run_experiments(exp_fun, parallel=PARALLEL):
    if parallel:
        p = multiprocessing.Pool(PARALLEL_POOL_SIZE)
        avg_res = p.map(exp_fun, range(ITERATIONS))
    else:
        avg_res = [exp_fun(i) for i in range(ITERATIONS)]

    print('Run experiments: %d results' % len(avg_res))


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    warnings.filterwarnings('ignore')
    b_time = time.time()
    run_experiments(exp_run_industrial_datasets, parallel=PARALLEL)
    print("Time costs: {:.3f}s".format(time.time() - b_time))

    visualize(p='add')
    print("All finished")
