import os
import pandas as pd
import glob

USE_LATEX = False
DATA_DIR = './results'
FIGURE_DIR = './results_csv'


def load_stats_dataframe(files, aggregated_results=None):
    if os.path.exists(aggregated_results) and all(
            [os.path.getmtime(f) < os.path.getmtime(aggregated_results) for f in files]):
        return pd.read_pickle(aggregated_results)

    df = pd.DataFrame()

    for f in files:
        print(f)
        tmp_dict = pd.read_pickle(f)

        tmp_dict['iteration'] = f.split('_')[-2]

        del tmp_dict['result']

        tmp_df = pd.DataFrame.from_dict(tmp_dict)
        df = pd.concat([df, tmp_df])

    if aggregated_results:
        df.to_pickle(aggregated_results)

    return df


def visualize(p):
    search_pattern = 'rq_*_stats.p'
    filename = p + '_rq'

    iteration_results = glob.glob(os.path.join(DATA_DIR, search_pattern))
    aggregated_results = os.path.join(FIGURE_DIR, filename)

    df = load_stats_dataframe(iteration_results, aggregated_results)

    pure_df = df[(df['detected'] + df['missed'] > 0)]
    mmm_df = pure_df.groupby(['env', 'rewardfun', 'agent'], as_index=False).mean(numeric_only=True)
    mmm_df.to_csv(os.path.join(FIGURE_DIR, p + '_result.csv'))
