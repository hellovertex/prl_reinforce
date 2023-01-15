import numpy as np
from enum import IntEnum


# Define MDP - State Space 
class S(IntEnum):
    # Non-Terminal States where player one has to act
    J = 0  # either JQ or JK -- Hero only sees his Jack
    Q = 1  # either QJ or QK -- Hero only sees his Queen
    K = 2  # either KJ or KQ -- Hero only sees his King
    JP_ = 3  # Jack  | Passed and villain Bet -- Hero has to act (Fold or Call)
    QP_ = 4  # Queen | Passed and villain Bet -- Hero has to act (Fold or Call)
    KP_ = 5  # King  | Passed and villain Bet -- Hero has to act (Fold or Call)

    # Terminal states where hero held Jack
    JPLQ = 6  # Jack | Passed | Lost vs. Passing Queen
    JPLK = 7  # Jack | Passed | Lost vs. Passing King
    JPx2 = 8  # Jack | Passed twice | Lost by default
    JPCQ = 9  # Jack | Passed and Called vs. betting Queen -- Lost
    JPCK = 10  # Jack | Passed and Called vs. betting King -- Lost
    JBWF = 11  # Jack | Bet | Won because villain folded
    JBSQ = 12  # Jack | Bet | Showdown vs. Calling Queen -- Lost
    JBSK = 13  # Jack | Bet | Showdown vs. Calling King -- Lost

    # Terminal states where hero held Queen
    QPWJ = 14  # Queen | Passed | Won vs. Passing Jack
    QPLK = 15  # Queen | Passed | Lost vs. Passing King
    QPx2 = 16  # Queen | Passed twice | Lost by default
    QPCJ = 17  # Queen | Passed and Called vs. betting Jack -- Won
    QPCK = 18  # Queen | Passed and Called vs. betting King -- Lost
    QBWF = 19  # Queen | Bet | Won because villain folded
    QBSJ = 20  # Queen | Bet | Showdown vs. Calling Jack -- Won
    QBSK = 21  # Queen | Bet | Showdown vs. Calling King -- Lost

    # Terminal states where hero held King
    KPWJ = 22  # King | Passed | Won vs. Passing Jack
    KPWQ = 23  # King | Passed | Won vs. Passing Queen
    KPx2 = 24  # King | Passed twice | Lost by default
    KPCJ = 25  # King | Passed and Called vs. betting Jack -- Won
    KPCQ = 26  # King | Passed and Called vs. betting Queen -- Won
    KBWF = 27  # King | Bet | Won because villain folded
    KBSJ = 28  # King | Bet | Showdown vs. Calling Jack -- Won
    KBSQ = 29  # King | Bet | Showdown vs. Calling Queen -- Won

    # Total number of terminal states
    N_STATESS = 30


J = S.J
Q = S.J
K = S.K


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
N_STATES = S.N_STATESS
N_ACTIONS = A.N_ACTIONS
T = np.zeros((N_STATES, N_ACTIONS, N_STATES))

# Assume villain has the following strategy: villain...
# 1) Always bets when facing a check, unless he has J (Villain only checks when holding J)
# 2) Always calls when facing a bet, unless he has J (Villain only folds when holding J).

# Rollouts where Hero holds Jack
T[J][check_fold][S.JP_] = 1  # villain will always bet when holding Q or K
T[J][bet_call][S.JBSQ] = .5  # Hero bets his J and gets called to showdown by Q -- Hero loses
T[J][bet_call][S.JBSK] = .5  # Hero bets his J and gets called to showdown by K -- Hero loses
T[S.JP_][check_fold][S.JPx2] = 1  # Checks and then folds after villain bets his Q or K -- loses 1
T[S.JP_][bet_call][S.JPCQ] = .5  # Checks and then calls villain betting his Q -- Hero loses 2
T[S.JP_][bet_call][S.JPCK] = .5  # Checks and then calls villain betting his K -- Hero loses 2
# Rollouts where hero holds Q
T[Q][check_fold][S.QPWJ] = .5  # Villain holds J and checks -- Hero wins by default
T[Q][check_fold][S.QP_] = .5  # Villain holds K and bets -- Hero has to move
T[Q][bet_call][S.QBWF] = .5  # Villain holds J and folds after Hero Bet -- Hero wins
T[Q][bet_call][S.QBSK] = .5  # Hero bets his Q and gets called to showdown by K -- Hero loses
T[S.QP_][check_fold][S.QPx2] = 1  # Checks and then folds after villain bets -- Hero loses 1
T[S.QP_][bet_call][S.QPCK] = 1  # Checks and then calls villain betting his K -- Hero loses 2
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
R[J][bet_call][S.JPCQ] = -2  # Hero checks and then calls villain raise
R[J][bet_call][S.JPCK] = -2  # Hero checks and then calls villain raise
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


def Qvalue_iteration(T, R, gamma=0.5, n_iters=10):
    Q = np.zeros((N_STATES, N_ACTIONS))
    for _ in range(n_iters):
        for s in range(N_STATES):  # for all states s
            for a in range(N_ACTIONS):  # for all actions a
                sum_sp = 0
                for S in range(N_STATES):  # for all reachable states s'
                    sum_sp += (T[s][a][S] * (R[s][a][S] + gamma * max(Q[S])))
                Q[s][a] = sum_sp
    return Q


if __name__ == "__main__":
    Q = Qvalue_iteration(T, R, 1, n_iters=1000)
    print(Q[0][0])
    print(Q[0][1])
