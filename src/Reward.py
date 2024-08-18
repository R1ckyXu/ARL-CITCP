import numpy as np


def TCF(result, sc):
    ordered_rewards = []
    if result[0] == 0:
        for tc in sc.testcases():
            ordered_rewards.append(0.0)
        return ordered_rewards

    total = result[0]

    rank_idx = np.array(result[-1]) - 1
    no_scheduled = len(sc.scheduled_testcases)

    rewards = np.zeros(no_scheduled)
    rewards[rank_idx] = 1

    for tc in sc.testcases():
        try:
            idx = sc.scheduled_testcases.index(tc)
            ordered_rewards.append(rewards[idx])
        except ValueError:
            ordered_rewards.append(0.0)  # Unscheduled test case
    return ordered_rewards


# Define the global interval number
INTVAL = 2


def NV_ALL_INT5(sc, bonus, intval=INTVAL):
    novlty_rewards = []
    for tc in sc.testcases():
        hisresults = tc['LastResults']
        intervalCycle = tc['intervalCycle']
        novlty = 0
        if len(hisresults) < 5 or intervalCycle >= intval:  # novlty
            novlty = bonus
        novlty_rewards.append(novlty)
    return novlty_rewards


def ATCF5(result, sc):
    ordered = TCF(result, sc)
    ordered_rewards = [2 * tc for tc in ordered]

    bonus = NV_ALL_INT5(sc=sc, bonus=2)

    assert len(bonus) == len(ordered_rewards), "len(bonus) ！= len(ordered_rewards)"

    for i in range(len(bonus)):
        bonus[i] += ordered_rewards[i]

    return bonus
