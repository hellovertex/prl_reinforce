import numpy as np
from enum import IntEnum


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


class A(IntEnum):
    # go left in game tree
    check_fold = 0  # actually "Pass" but pass is reserved python keyword, so call it "check_fold" here
    # go right in game tree
    bet_call = 1
    N_ACTIONS = 2


# Initialize transition function
N_NON_TERMINAL_STATES = S.N_STATES_S
N_ACTIONS = A.N_ACTIONS
N_TERMINAL_STATES = S_.N_STATES_S_

T = np.zeros((N_NON_TERMINAL_STATES, N_ACTIONS, N_TERMINAL_STATES))
T[S.J][A.check_fold][S_.JPLQ] = 1/2
T[S.J][A.check_fold][S_.JPLK] = 0  # Villain never passes when having King
T[S.J][A.check_fold][S.JP_] = 1/2


def Qvalue_iteration(T, R, gamma=0.5, n_iters=10):
    nA = R.shape[0]
    nS = T.shape[0]
    Q = np.zeros((nS, nA))  # initially
    for _ in range(n_iters):
        for s in range(nS):  # for all states s
            for a in range(nA):  # for all actions a
                sum_sp = 0
                for s_ in range(nS):  # for all reachable states s'
                    sum_sp += (T[s][a][s_] * (R[s][a][s_] + gamma * max(Q[s_])))
                Q[s][a] = sum_sp
    return Q


if __name__ == "__main__":
    Q = np.zeros((S.N_STATES_S, S_.N_STATES_S_))
