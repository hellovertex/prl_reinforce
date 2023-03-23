from enum import auto, IntEnum
import random
from typing import List, Dict

from q_value_iteration import S, N_ACTIONS, N_STATES


class Card(IntEnum):
    J = 0
    Q = 1
    K = 2


class Action(IntEnum):
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


S = State


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
    s_4 = [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]
    s_5 = [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    s_6 = [1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0]
    s_7 = [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    s_8 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    s_9 = [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    s_10 = [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
    s_11 = [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]
    s_12 = [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    s_13 = [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]
    s_14 = [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]
    s_15 = [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
    s_16 = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    s_17 = [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    s_18 = [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    s_19 = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
    s_20 = [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]
    s_21 = [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
    s_22 = [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]
    s_23 = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
    s_24 = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    s_25 = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    s_26 = [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
    s_27 = [0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]
    s_28 = [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    s_29 = [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]
    s_30 = [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]
    s_31 = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
    s_32 = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    s_33 = [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    s_34 = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    s_35 = [0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]
    s_36 = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
    s_37 = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1]
    s_38 = [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
    s_39 = [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0]
    s_40 = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    s_41 = [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    s_42 = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    s_43 = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
    s_44 = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0]
    s_45 = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    s_46 = [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0]
    s_47 = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]


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


class KuhnPokerEnvironment:
    """This class provides an interface for an agent to interact with
    the underlying MDP of Kuhn Poker"""

    def __init__(self):
        self._player_cards: Dict[Players, Card] = dict()
        self._action_history: List[Action] = list()
        self._deck: List[Card] = list()
        self._state = None

    def draw_card(self):
        assert len(self._deck) > 0
        card = random.choice(self._deck)
        self._deck.pop(card)
        return card

    def vectorize(self):
        # one hot encodings
        cards = [0 for _ in range(len(Card) * len(Players))]
        cards[self._player_cards[Players.one]] = 1
        cards[len(Card) + self._player_cards[Players.two]] = 1

        actions = [0 for _ in range(len(Action) * len(Stage))]
        for i, action in enumerate(self._action_history):
            actions[i * len(Stage) + action.value] = 1
        # current player is implicitly determined from action
        # current player might also be explicitly encoded but
        # its just overhead to keep track of
        return cards + actions

    def reset(self):
        self._player_cards[Players.one] = self.draw_card()
        self._player_cards[Players.two] = self.draw_card()
        self.state = self.vectorize()
        return self.state

    def t(self, s, a, s_, hero_strategy, villain_strategy) -> float:
        if self.state[S.action_0_R + S.action_0_L] == 0:
            # a is first action
            new_state = s
            new_state[State.action_0_L + a] = 1
            if s_ == new_state:
                return 1
            else:
                return 0
        elif self.state[S.action_1_R + S.action_1_L] == 0:
            # a is second action
            pass
        elif self.state[S.action_2_R + S.action_2_L] == 0:
            # a is third action
            pass
        else:
            return 0

    def r(self, s, a, s_):
        pass

    def step(self, action: int):
        """
        :param action: Integer representation of the action to execute in the environment.
        :return: state, reward, done, info
        """
        # if self.state[S.action_0_R+S.action_0_L] == 0:
        #     pass
        # elif self.state[S.action_1_R+S.action_1_L] == 0:
        #     pass
        # elif self.state[S.action_2_R+S.action_2_L] == 0:
        #     pass
        # else:
        #     return
        pass


if __name__ == '__main__':
    _ = [0, 0, 0, 0, 0, 0]
    l = [1, 0, 0, 0, 0, 0]
    r = [0, 1, 0, 0, 0, 0]
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
        for p in [_, l, r, ll, lrl, lrr, rl, rr]:
            states.append(s + p)
    for i, state in enumerate(states):
        print(f's_{i} = {state}')
