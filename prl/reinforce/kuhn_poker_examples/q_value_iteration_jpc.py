# todo: design h1,h2,v1,v2 from qlearning
#  plot q vals
from enum import IntEnum
from functools import partial

import numpy as np
import pandas as pd


class Card(IntEnum):
    J = 0
    Q = 1
    K = 2


class Actions(IntEnum):
    _pass = 0
    _bet = 1


class Players(IntEnum):
    one = 0
    two = 1


class Stage(IntEnum):
    one = 0
    two = 1
    three = 2


class State(IntEnum):
    p0_has_J = 0
    p0_has_Q = 1
    p0_has_K = 2
    p1_has_J = 3
    p1_has_Q = 4
    p1_has_K = 5

    action_0_L = 6
    action_0_R = 7
    action_1_L = 8
    action_1_R = 9
    action_2_L = 10
    action_2_R = 11


class States:
    s_0 = [
        1, 0, 0,  # Player 1 Card is J
        0, 1, 0,  # Player 2 Card is Q
        0, 0,  # Player 1 1st. Action L=0, Action R=0
        0, 0,  # Player 2 Action L=0, Action R=0
        0, 0  # Player 1 2nd. Action L=0, Action R=0
    ]
    s_1 = [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    s_2 = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    s_3 = [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0]
    s_4 = [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    s_5 = [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]
    s_6 = [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    s_7 = [1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0]
    s_8 = [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    s_9 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    s_10 = [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    s_11 = [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
    s_12 = [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]
    s_13 = [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
    s_14 = [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    s_15 = [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]
    s_16 = [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]
    s_17 = [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
    s_18 = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    s_19 = [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    s_20 = [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    s_21 = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
    s_22 = [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    s_23 = [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]
    s_24 = [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
    s_25 = [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]
    s_26 = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
    s_27 = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    s_28 = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    s_29 = [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
    s_30 = [0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]
    s_31 = [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
    s_32 = [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    s_33 = [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]
    s_34 = [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]
    s_35 = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
    s_36 = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    s_37 = [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    s_38 = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    s_39 = [0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]
    s_40 = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    s_41 = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
    s_42 = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1]
    s_43 = [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
    s_44 = [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0]
    s_45 = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    s_46 = [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    s_47 = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    s_48 = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
    s_49 = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    s_50 = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0]
    s_51 = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    s_52 = [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0]
    s_53 = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    as_list = [
        s_0,
        s_1,
        s_2,
        s_3,
        s_4,
        s_5,
        s_6,
        s_7,
        s_8,
        s_9,
        s_10,
        s_11,
        s_12,
        s_13,
        s_14,
        s_15,
        s_16,
        s_17,
        s_18,
        s_19,
        s_20,
        s_21,
        s_22,
        s_23,
        s_24,
        s_25,
        s_26,
        s_27,
        s_28,
        s_29,
        s_30,
        s_31,
        s_32,
        s_33,
        s_34,
        s_35,
        s_36,
        s_37,
        s_38,
        s_39,
        s_40,
        s_41,
        s_42,
        s_43,
        s_44,
        s_45,
        s_46,
        s_47,
        s_48,
        s_49,
        s_50,
        s_51,
        s_52,
        s_53
    ]


TERMINAL_STATES = [
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
]

S = State
N_STATES = 54
N_ACTIONS = 2


class PlayerStrategy(IntEnum):
    J_L0 = 0
    J_R0 = 1
    J_L1 = 2
    J_R1 = 3

    Q_L0 = 4
    Q_R0 = 5
    Q_L1 = 6
    Q_R1 = 7

    K_L0 = 8
    K_R0 = 9
    K_L1 = 10
    K_R1 = 11


def cards_match(s, s_):
    return np.array_equal(s[:len(Card) * len(Players)], s_[:len(Card) * len(Players)])


def plot_q_values(Q: np.ndarray):
    import seaborn as sns
    import matplotlib.pylab as plt
    plt.style.use("seaborn")
    plt.set_cmap('viridis')
    states = ['J_Action_1',
              'Q_Action_1',
              'K_Action_1',
              'J_Action_2',
              'Q_Action_2',
              'K_Action_2',
              ]
    # plt.figure(figsize=(6,2))
    df = pd.DataFrame(Q[:6, ].T)  # , columns=state_names[:6])
    ax = sns.heatmap(df,
                     linewidth=1,
                     annot=True,
                     # xticklabels=True,
                     # yticklabels=True
                     yticklabels=['Check or Fold', 'Bet or Call'],
                     xticklabels=states
                     )
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # plt.xticks(rotation=10)
    # ax.tick_params(axis='x', which='major', pad=0)
    # b, t = plt.ylim()  # discover the values for bottom and top
    # b += 0.5  # Add 0.5 to the bottom
    # t -= 0.5  # Subtract 0.5 from the top
    # plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.title("Kuhn-Poker: Q-Values of states where Player One has to act",
              fontweight='bold', pad=20)
    plt.show()


def apply_strategy(strategy, state):
    # get player who has to act
    first_move_is_done = state[S.action_0_R] + state[S.action_0_L]
    second_move_is_done = state[S.action_1_R] + state[S.action_1_L]
    if bool(first_move_is_done):
        # villain moves
        card = np.where(state[State.p1_has_J:State.p1_has_K + 1])[0][0]
        first_move = np.where(state[State.action_0_L:State.action_0_R + 1])[0][0]
        prob_l = strategy[4 * card + 2 * first_move]
        prob_r = strategy[4 * card + 2 * first_move + 1]
    else:
        card = np.where(state[State.p0_has_J:State.p0_has_K + 1])[0][0]
        # hero moves
        prob_l = strategy[4 * card + 2 * second_move_is_done]
        prob_r = strategy[4 * card + 2 * second_move_is_done + 1]
    return prob_l, prob_r


def apply_action(state, action):
    next_state = np.zeros_like(state)
    next_state[np.where(state)[0]] = 1
    first_move_is_done = state[S.action_0_R] + state[S.action_0_L]
    second_move_is_done = state[S.action_1_R] + state[S.action_1_L]
    third_move_is_done = state[S.action_2_R] + state[S.action_2_L]
    if bool(third_move_is_done):
        return np.zeros_like(state)
    if not first_move_is_done:
        next_state[S.action_0_L + action] = 1
    elif first_move_is_done:
        next_state[S.action_1_L + action] = 1
    elif second_move_is_done:
        next_state[S.action_2_L + action] = 1
    else:
        raise ValueError(f"Edge case encountered when trying to step "
                         f"state={state} with action={action}")
    return next_state


def t(s, a, s_, hero_strategy, villain_strategy) -> float:
    # check if s_ is reachable from s given action a
    if s == s_:
        return 0
    next_state = apply_action(s, a)
    if not np.array_equal(next_state, s_):
        return 0

    # get player who has to act
    first_move_is_done = bool(s[S.action_0_R] + s[S.action_0_L])
    second_move_is_done = bool(s[S.action_1_R] + s[S.action_1_L])
    third_move_is_done = bool(s[S.action_2_R] + s[S.action_2_L])
    if third_move_is_done:
        return 0
    if first_move_is_done:
        if second_move_is_done:
            # apply hero second move get [prob(L), prob(R)]
            probs = apply_strategy(hero_strategy, s)
            return probs[a]
        else:
            # apply villan first move get [prob(L), prob(R)]
            # compute each succ state and if one matches s_ return that prob
            probs = apply_strategy(villain_strategy, s)
            return probs[a]
    else:
        # apply hero first move get [prob(L), prob(R)]
        # compute each succ state and if one matches s_ return that prob
        probs = apply_strategy(hero_strategy, s)
        return probs[a]


def r(s, a, s_):
    if np.array_equal(s, s_):
        return 0
    if s_ not in TERMINAL_STATES:
        return 0
    card = np.where(s_[State.p0_has_J:State.p0_has_K + 1])[0][0]
    hero_wins = np.where(s_[State.p1_has_J:State.p1_has_K + 1])[0][0] < card
    if s_[State.action_0_R] and s_[State.action_1_R]:
        return 2 if hero_wins else -2
    if s_[State.action_0_L] and s_[State.action_1_R] and s_[State.action_2_R]:
        return 2 if hero_wins else -2
    return 1 if hero_wins else -1


def util_make_states():
    _ = [0, 0, 0, 0, 0, 0]
    l = [1, 0, 0, 0, 0, 0]
    r = [0, 1, 0, 0, 0, 0]
    ll = [1, 0, 1, 0, 0, 0]
    lr = [1, 0, 0, 1, 0, 0]
    lrl = [1, 0, 0, 1, 1, 0]
    lrr = [1, 0, 0, 1, 0, 1]
    rl = [0, 1, 1, 0, 0, 0]
    rr = [0, 1, 0, 1, 0, 0]
    cards = []
    for i in range(len(Card)):
        for j in range(len(Card)):
            if i != j:
                s = [0, 0, 0, 0, 0, 0]
                s[i] = 1
                s[j + 3] = 1
                cards.append(s)

    states = []
    for s in cards:
        for p in [_, l, r, ll, lr, lrl, lrr, rl, rr]:
            states.append(s + p)
    for i, state in enumerate(states):
        print(f's_{i} = {state}')


def util_make_terminal_states():
    ll = [1, 0, 1, 0, 0, 0]
    lrl = [1, 0, 0, 1, 1, 0]
    lrr = [1, 0, 0, 1, 0, 1]
    rl = [0, 1, 1, 0, 0, 0]
    rr = [0, 1, 0, 1, 0, 0]
    cards = []
    for i in range(len(Card)):
        for j in range(len(Card)):
            if i != j:
                s = [0, 0, 0, 0, 0, 0]
                s[i] = 1
                s[j + 3] = 1
                cards.append(s)

    states = []
    for s in cards:
        for p in [ll, lrl, lrr, rl, rr]:
            states.append(s + p)
    for i, state in enumerate(states):
        print(f's_{i} = {state}')


def Qvalue_iteration(t, r, gamma=0.5, n_iters=10):
    Q = np.zeros((N_STATES, N_ACTIONS))
    for i in range(n_iters):
        # df = export_q_values(Q, path=f'./exported_q_values_{i}.csv')
        # print(df.head())
        for i, s in enumerate(States.as_list):  # for all states s
            for i, a in enumerate(list(Actions)):  # for all actions a
                qs = 0
                for j, s_ in enumerate(States.as_list):  # for all reachable states s'
                    qs += (t(s, a, s_) * (r(s, a, s_) + gamma * max(Q[j])))
                Q[i][a] = qs
    # plot_q_values(Q)
    return Q


if __name__ == '__main__':
    # util_make_states()
    # util_make_terminal_states()
    hero_strategy = [
        # JACK
        1, 0,  # pass first action with probability 1
        1, 0,  # pass after p2 bet with probability 1
        # QUEEN
        1, 0,  # pass first action with probability 1
        1, 0,  # pass after p2 bet with probability 1
        # KING
        0, 1,  # bet first action with probability 1
        0, 1  # call after p2 bet with probability 1
    ]
    villain_strategy = [
        # JACK
        1, 0,  # pass after p1 checked with probability 1
        1, 0,  # pass after p1 bet with probability 1
        # QUEEN
        0, 1,  # bet after p1 checked with probability 1
        0, 1,  # call after p1 bet with probability 1
        # KING
        0, 1,  # bet after p1 checked with probability 1
        0, 1  # call after p1 bet with probability 1
    ]
    t_fn = partial(t,
                   hero_strategy=hero_strategy,
                   villain_strategy=villain_strategy)
    Q = Qvalue_iteration(t_fn, r, 1, n_iters=10)
    print(Q)
