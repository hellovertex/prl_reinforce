from enum import auto, IntEnum
import random
from typing import List, Dict

import numpy as np

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
