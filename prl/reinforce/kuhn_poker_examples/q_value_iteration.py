import numpy as np
from enum import IntEnum, auto


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
    K_P1_bet_P2_call_show_J = 29
    K_P1_bet_P2_call_show_Q = 30


J = S.J____
Q = S.J____
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
T[Q][bet_call][S.] = .5  # Villain holds J and folds after Hero Bet -- Hero wins
T[Q][bet_call][S.QBSK] = .5  # Hero bets his Q and gets called to showdown by K -- Hero loses
T[S.Q_P1_check_P2_bet_P1_PENDING][check_fold][S.QPx2] = 1  # Checks and then folds after villain bets -- Hero loses 1
T[S.Q_P1_check_P2_bet_P1_PENDING][bet_call][S.QPCK] = 1  # Checks and then calls villain betting his K -- Hero loses 2
# Rollouts where hero holds K
T[K][check_fold][S.KPWJ] = .5  # Villain holds J and checks -- Hero wins by default
T[K][check_fold][S.KP_] = .5  # Villain holds Q and bets -- Hero has to move
T[K][bet_call][S.KBWF] = .5  # Villain holds J and folds after Hero Bet -- Hero wins
T[K][bet_call][S.KBSQ] = .5  # Hero bets his Q and gets called to showdown by Q -- Hero wins
T[S.KP_][check_fold][S.KPCQ] = 1  # Checks and then folds after villain bets -- Hero loses 1 by default
T[S.KP_][bet_call][S.KPCQ] = 1  # Checks and then calls after villain bets his Q -- Hero wins 2

# Define MDP - Reward Function
R = np.zeros((N_STATES, N_ACTIONS, N_STATES))
R[J][check_fold][S.JP_] = 0  # S.JP_ is reached with probability 1 because villain always bets Q, K
R[J][bet_call][S.JBSQ] = -2  # Hero bets and then gets a call from villain with Q -- loses 2
R[J][bet_call][S.JBSK] = -2  # Hero bets and then gets a call from villain with K -- loses 2
R[S.JP_][check_fold][S.JPx2] = -1  # Hero checks and then folds and loses 1 by default
R[S.JP_][bet_call][S.JPCQ] = -2  # Hero checks and then calls and loses 2
R[S.JP_][bet_call][S.JPCQ] = -2  # Hero checks and then calls and loses 2

R[Q][check_fold][S.QP_] = 0  # Hero has to act
R[Q][check_fold][S.QPWJ] = 1  # Hero wins by default
R[Q][bet_call][S.QBWF] = 1  # Villain folds his J
R[Q][bet_call][S.QBSK] = -2  # Villain raises his K
R[S.QP_][check_fold][S.QPx2] = -1  # Hero loses ante=1 after check/fold
R[S.QP_][bet_call][S.QPCK] = -2  # Hero passes then calls bet of villain with King

R[K][check_fold][S.KPWJ] = 1  # Villain holds J and checks -- Hero wins by default
R[K][check_fold][S.KP_] = 0  # Villain holds Q and bets -- Hero has to move
R[K][bet_call][S.KBWF] = 1  # Villain holds J and folds after Hero Bet -- Hero wins
R[K][bet_call][S.KBSQ] = 2  # Hero bets his Q and gets called to showdown by Q -- Hero wins
R[S.KP_][check_fold][S.KPCQ] = -1  # Checks and then folds after villain bets -- Hero loses 1 by default
R[S.KP_][bet_call][S.KPCQ] = 2  # Checks and then calls after villain bets his Q -- Hero wins 2

import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame()  # fill with Q values


# todo make surface plot using https://plotly.com/python/3d-surface-plots/
#  and debug below function

def Qvalue_iteration(T, R, gamma=0.5, n_iters=10):
    Q = np.zeros((N_STATES, N_ACTIONS))
    for _ in range(n_iters):
        for s in range(N_STATES):  # for all states s
            for a in range(N_ACTIONS):  # for all actions a
                sum_sp = 0
                for S in range(N_STATES):  # for all reachable states s'
                    sum_sp += (T[s][a][S] * (R[s][a][S] + gamma * max(Q[S])))
                Q[s][a] = sum_sp
        # dataframe after every iteration
        df = pd.DataFrame(Q.T, columns=state_names[:-1])
        print(df.head())

    return Q


if __name__ == "__main__":
    Q = Qvalue_iteration(T, R, 1, n_iters=10)
    print(Q[0][0])
    print(Q[0][1])
