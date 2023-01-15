import numpy as np
from enum import IntEnum


# Define MDP - State Space I
class S(IntEnum):
    # Non-Terminal States where player one has to act
    J: 0  # either JQ or JK -- Hero only sees his Jack
    Q: 1  # either QJ or QK -- Hero only sees his Queen
    K: 2  # either KJ or KQ -- Hero only sees his King
    JP_: 3  # Jack  | Passed and villain Bet -- Hero has to act (Fold or Call)
    QP_: 4  # Queen | Passed and villain Bet -- Hero has to act (Fold or Call)
    KP_: 5  # King  | Passed and villain Bet -- Hero has to act (Fold or Call)

    # Total number of states where action is required
    N_STATES_S: 6


J = S.J
Q = S.J
K = S.K


# Define MDP - State Space II
class S_(IntEnum):
    # Terminal states where hero held Jack
    JPLQ: 0  # Jack | Passed | Lost vs. Passing Queen
    JPLK: 1  # Jack | Passed | Lost vs. Passing King
    JPx2: 2  # Jack | Passed twice | Lost by default
    JPCQ: 3  # Jack | Passed and Called vs. betting Queen -- Lost
    JPCK: 4  # Jack | Passed and Called vs. betting King -- Lost
    JBWF: 5  # Jack | Bet | Won because villain folded
    JBSQ: 6  # Jack | Bet | Showdown vs. Calling Queen -- Lost
    JBSK: 7  # Jack | Bet | Showdown vs. Calling King -- Lost

    # Terminal states where hero held Queen
    QPWJ: 8  # Queen | Passed | Won vs. Passing Jack
    QPLK: 9  # Queen | Passed | Lost vs. Passing King
    QPx2: 10  # Queen | Passed twice | Lost by default
    QPCJ: 11  # Queen | Passed and Called vs. betting Jack -- Won
    QPCK: 12  # Queen | Passed and Called vs. betting King -- Lost
    QBWF: 13  # Queen | Bet | Won because villain folded
    QBSJ: 14  # Queen | Bet | Showdown vs. Calling Jack -- Won
    QBSK: 15  # Queen | Bet | Showdown vs. Calling King -- Lost

    # Terminal states where hero held King
    KPWJ: 16  # King | Passed | Won vs. Passing Jack
    KPWQ: 17  # King | Passed | Won vs. Passing Queen
    KPx2: 18  # King | Passed twice | Lost by default
    KPCJ: 19  # King | Passed and Called vs. betting Jack -- Won
    KPCQ: 20  # King | Passed and Called vs. betting Queen -- Won
    KBWF: 21  # King | Bet | Won because villain folded
    KBSJ: 22  # King | Bet | Showdown vs. Calling Jack -- Won
    KBSQ: 23  # King | Bet | Showdown vs. Calling Queen -- Won

    # Repeat non-terminal states from set S here, which are in the set of successor states
    JP_: 24
    QP_: 25
    KP_: 26

    # Total number of terminal states
    N_STATES_S_: 27


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
N_STATES = S.N_STATES_S
N_ACTIONS = A.N_ACTIONS
N_SUCCESSOR_STATES = S_.N_STATES_S_
T = np.zeros((N_STATES, N_ACTIONS, N_SUCCESSOR_STATES))

# Assume villain has the following strategy: villain...
# 1) Always bets when facing a check, unless he has J (Villain only checks when holding J)
# 2) Always calls when facing a bet, unless he has J (Villain only folds when holding J).

# Rollouts where Hero holds Jack
T[J][check_fold][S_.JP_] = 1  # villain will always bet when holding Q or K
T[J][bet_call][S_.JBSQ] = .5  # Hero bets his J and gets called to showdown by Q -- Hero loses
T[J][bet_call][S_.JBSK] = .5  # Hero bets his J and gets called to showdown by K -- Hero loses
T[S.JP_][check_fold][S_.JPx2] = 1  # Checks and then folds after villain bets his Q or K -- loses 1
T[S.JP_][bet_call][S_.JPCQ] = .5  # Checks and then calls villain betting his Q -- Hero loses 2
T[S.JP_][bet_call][S_.JPCK] = .5  # Checks and then calls villain betting his K -- Hero loses 2
# Rollouts where hero holds Q
T[Q][check_fold][S_.QPWJ] = .5  # Villain holds J and checks -- Hero wins by default
T[Q][check_fold][S_.QP_] = .5  # Villain holds K and bets -- Hero has to move
T[Q][bet_call][S_.QBWF] = .5  # Villain holds J and folds after Hero Bet -- Hero wins
T[Q][bet_call][S_.QBSK] = .5  # Hero bets his Q and gets called to showdown by K -- Hero loses
T[S.QP_][check_fold][S_.QPx2] = 1  # Checks and then folds after villain bets -- Hero loses 1
T[S.QP_][bet_call][S_.QPCK] = 1  # Checks and then calls villain betting his K -- Hero loses 2
# Rollouts where hero holds K
T[K][check_fold][S_.KPWJ] = .5  # Villain holds J and checks -- Hero wins by default
T[K][check_fold][S_.KP_] = .5  # Villain holds Q and bets -- Hero has to move
T[K][bet_call][S_.KBWF] = .5  # Villain holds J and folds after Hero Bet -- Hero wins
T[K][bet_call][S_.KBSQ] = .5  # Hero bets his Q and gets called to showdown by Q -- Hero wins
T[S.KP_][check_fold][S_.KPCQ] = 1  # Checks and then folds after villain bets -- Hero loses 1 by default
T[S.KP_][bet_call][S_.KPCQ] = 1  # Checks and then calls after villain bets his Q -- Hero wins 2

# Define MDP - Reward Function
R = np.zeros((N_STATES, N_ACTIONS, N_SUCCESSOR_STATES))
R[J][check_fold][S_.JP_] = 0  # S_.JP_ is reached with probability 1 because villain always bets Q, K
R[J][bet_call][S_.JPCQ] = -2  # Hero checks and then calls villain raise
R[J][bet_call][S_.JPCK] = -2  # Hero checks and then calls villain raise
R[S.JP_][check_fold][S_.JPx2] = -1  # Hero checks and then folds and loses 1 by default
R[S.JP_][bet_call][S_.JPCQ] = -2  # Hero checks and then calls and loses 2
R[S.JP_][bet_call][S_.JPCQ] = -2  # Hero checks and then calls and loses 2

R[Q][check_fold][S_.QP_] = 0  # Hero has to act
R[Q][check_fold][S_.QPWJ] = 1  # Hero wins by default
R[Q][bet_call][S_.QBWF] = 1  # Villain folds his J
R[Q][bet_call][S_.QBSK] = -2  # Villain raises his K
R[S.QP_][check_fold][S_.QPx2] = -1  # Hero loses ante=1 after check/fold
R[S.QP_][bet_call][S_.QPCK] = -2  # Hero passes then calls bet of villain with King

R[K][check_fold][S_.KPWJ] = 1  # Villain holds J and checks -- Hero wins by default
R[K][check_fold][S_.KP_] = 0  # Villain holds Q and bets -- Hero has to move
R[K][bet_call][S_.KBWF] = 1  # Villain holds J and folds after Hero Bet -- Hero wins
R[K][bet_call][S_.KBSQ] = 2  # Hero bets his Q and gets called to showdown by Q -- Hero wins
R[S.KP_][check_fold][S_.KPCQ] = -1  # Checks and then folds after villain bets -- Hero loses 1 by default
R[S.KP_][bet_call][S_.KPCQ] = 2  # Checks and then calls after villain bets his Q -- Hero wins 2


def Qvalue_iteration(T, R, gamma=0.5, n_iters=10):
    Q = np.zeros((N_STATES, N_ACTIONS))
    for _ in range(n_iters):
        for s in range(N_STATES):  # for all states s
            for a in range(N_ACTIONS):  # for all actions a
                sum_sp = 0
                for s_ in range(N_SUCCESSOR_STATES):  # for all reachable states s'
                    sum_sp += (T[s][a][s_] * (R[s][a][s_] + gamma * max(Q[s_])))
                Q[s][a] = sum_sp
    return Q


if __name__ == "__main__":
