from enum import IntEnum
from typing import Type

import numpy as np


# Define MDP - State Space
class S(IntEnum):
    # Non-Terminal States where player one has to act
    J____ = 0
    Q____ = 1
    K____ = 2
    J_P1_check_P2_bet_P1_PENDING = 3
    Q_P1_check_P2_bet_P1_PENDING = 4
    K_P1_check_P2_bet_P1_PENDING = 5

    # Terminal states where hero held Jack
    J_P1_check_P2_check_show_Q = 6
    J_P1_check_P2_check_show_K = 7
    J_P1_check_P2_bet_P1_fold = 8
    J_P1_check_P2_bet_P1_call_P2_show_Q = 9
    J_P1_check_P2_bet_P1_call_P2_show_K = 10
    J_P1_bet_P2_fold = 11
    J_P1_bet_P2_call_show_Q = 12
    J_P1_bet_P2_call_show_K = 13

    # Terminal states where hero held Queen
    Q_P1_check_P2_check_show_J = 14
    Q_P1_check_P2_check_show_K = 15
    Q_P1_check_P2_bet_P1_fold = 16
    Q_P1_check_P2_bet_P1_call_P2_show_J = 17
    Q_P1_check_P2_bet_P1_call_P2_show_K = 18
    Q_P1_bet_P2_fold = 19
    Q_P1_bet_P2_call_show_J = 20
    Q_P1_bet_P2_call_show_K = 21

    # Terminal states where hero held King
    K_P1_check_P2_check_show_J = 22
    K_P1_check_P2_check_show_Q = 23
    K_P1_check_P2_bet_P1_fold = 24
    K_P1_check_P2_bet_P1_call_P2_show_J = 25
    K_P1_check_P2_bet_P1_call_P2_show_Q = 26
    K_P1_bet_P2_fold = 27
    K_P1_bet_P2_call_show_J = 28
    K_P1_bet_P2_call_show_Q = 29


J = S.J____
Q = S.Q____
K = S.K____
state_names = list(S.__members__.keys())


# Define MDP - Action Space
class A(IntEnum):
    # go left in game tree
    check_fold = 0  # actually "Pass" but pass is reserved python keyword, so call it "check_fold" here
    # go right in game tree
    bet_call = 1
    N_ACTIONS = 2


check_fold = A.check_fold
bet_call = A.bet_call

# Define MDP - Transition function
N_STATES = len(state_names)
N_ACTIONS = A.N_ACTIONS
T = np.zeros((N_STATES, N_ACTIONS, N_STATES))

# Assume villain has the following strategy: villain...
# 1) Always bets when facing a check, unless he has J (Villain only checks when holding J)
# 2) Always calls when facing a bet, unless he has J (Villain only folds when holding J).

# Rollouts where Hero holds Jack
# villain will always bet when holding Q or K
T[J][check_fold][S.J_P1_check_P2_bet_P1_PENDING] = 1
# Hero bets his J and gets called to showdown by Q -- Hero loses
T[J][bet_call][S.J_P1_bet_P2_call_show_Q] = .5
# Hero bets his J and gets called to showdown by K -- Hero loses
T[J][bet_call][S.J_P1_bet_P2_call_show_K] = .5
# Checks and then folds after villain bets his Q or K -- loses 1
T[S.J_P1_check_P2_bet_P1_PENDING][check_fold][S.J_P1_check_P2_bet_P1_fold] = 1
# Checks and then calls villain betting his Q -- Hero loses 2
T[S.J_P1_check_P2_bet_P1_PENDING][bet_call][S.J_P1_check_P2_bet_P1_call_P2_show_Q] = .5
# Checks and then calls villain betting his K -- Hero loses 2
T[S.J_P1_check_P2_bet_P1_PENDING][bet_call][S.J_P1_check_P2_bet_P1_call_P2_show_K] = .5

# Rollouts where hero holds Q
# Villain holds K and bets -- Hero has to move
T[Q][check_fold][S.Q_P1_check_P2_bet_P1_PENDING] = .5
# Villain holds J and checks -- Hero wins by default
T[Q][check_fold][S.Q_P1_check_P2_check_show_J] = .5
# Villain holds J and folds after Hero Bet -- Hero wins
T[Q][bet_call][S.Q_P1_bet_P2_fold] = .5
# Hero bets his Q and gets called to showdown by K -- Hero loses
T[Q][bet_call][S.Q_P1_bet_P2_call_show_K] = .5
# Checks and then folds after villain bets -- Hero loses 1
T[S.Q_P1_check_P2_bet_P1_PENDING][check_fold][S.Q_P1_check_P2_bet_P1_fold] = 1
# Checks and then calls villain betting his K -- Hero loses 2
T[S.Q_P1_check_P2_bet_P1_PENDING][bet_call][S.Q_P1_check_P2_bet_P1_call_P2_show_K] = 1

# Rollouts where hero holds K
# Villain holds J and checks -- Hero wins by default
T[K][check_fold][S.K_P1_check_P2_check_show_J] = .5
# Villain holds Q and bets -- Hero has to move
T[K][check_fold][S.K_P1_check_P2_bet_P1_PENDING] = .5
# Villain holds J and folds after Hero Bet -- Hero wins
T[K][bet_call][S.K_P1_bet_P2_fold] = .5
# Hero bets his K and gets called to showdown by Q -- Hero wins
T[K][bet_call][S.K_P1_bet_P2_call_show_Q] = .5
# Checks and then folds after villain bets -- Hero loses 1 by default
T[S.K_P1_check_P2_bet_P1_PENDING][check_fold][S.K_P1_check_P2_bet_P1_fold] = 1
# Checks and then calls after villain bets his Q -- Hero wins 2
T[S.K_P1_check_P2_bet_P1_PENDING][bet_call][S.K_P1_check_P2_bet_P1_call_P2_show_Q] = 1

cards = ['J', 'Q', 'K']


# Define MDP - Reward Function
def R(s: str, a, s_: Type[S]):
    """ Computes the reward function corresponding to the transition function
    of the Kuhn-Poker example above. In our case the reward is determined by s_ alone """
    # Parse state-index to state-name string representation
    s_ = S(s_).name
    # If initial state is successor state s_ -- all actions get reward of 0
    if '____' in s_:
        return 0
    # Player 1 checked, Player 2 bet, Player 1 needs to act
    if 'PENDING' in s_:
        return 0
    # Both players checked, win or lose ante, depending on card strength
    if 'P2_check' in s_:
        p1_card = s_[0]
        p2_card = s_[-1]
        return -1 if cards.index(p1_card) < cards.index(p2_card) else 1
    # If player one folds after a bet, he loses his ante, regardless of card strength
    if 'P2_bet_P1_fold' in s_:
        return -1
    # Player 1 checks and Player 2 bets, Player 1 then calls,
    # the better hand wins the other players ante
    if 'P2_bet_P1_call' in s_:
        p1_card = s_[0]
        p2_card = s_[-1]
        return -2 if cards.index(p1_card) < cards.index(p2_card) else 2
    # Player 1 bets and Player 2 folds in response
    if 'P2_fold' in s_:
        return 1
    # Player 1 bets and Player 2 calls, the better hand wins the other players ante
    if 'P2_call' in s_:
        p1_card = s_[0]
        p2_card = s_[-1]
        return -2 if cards.index(p1_card) < cards.index(p2_card) else 2
    raise ValueError(f'The state {s_} is missing in the definition of the State Space. '
                     f'Fix class S(IntEnum) defined above.')


import pandas as pd
import numpy as np

df = pd.DataFrame()  # fill with Q values


# todo make surface plot using https://plotly.com/python/3d-surface-plots/
#  and debug below function
def export_q_values(Q: np.ndarray, path='./exported_q_values.csv'):
    df = pd.DataFrame(Q.T, columns=state_names)
    df.to_csv(path)
    return df


def Qvalue_iteration(T, R, gamma=0.5, n_iters=10):
    Q = np.zeros((N_STATES, N_ACTIONS))
    for i in range(n_iters):
        # df = export_q_values(Q, path=f'./exported_q_values_{i}.csv')
        # print(df.head())
        for s in range(N_STATES):  # for all states s
            for a in range(N_ACTIONS):  # for all actions a
                sum_sp = 0
                for s_ in range(N_STATES):  # for all reachable states s'
                    r = R(s, a, s_)
                    max_q = max(Q[s_])
                    t = T[s][a][s_]
                    sum_sp += (t * (r + gamma * max_q))
                Q[s][a] = sum_sp
    return Q


if __name__ == "__main__":
    Q = Qvalue_iteration(T, R, 1, n_iters=3)
    print(Q)
