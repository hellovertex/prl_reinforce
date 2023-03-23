import enum
import random
from typing import List, Dict

from q_value_iteration import S, N_ACTIONS, N_STATES


class Card(enum.IntEnum):
    J = 0
    Q = 1
    K = 2


class Action(enum.IntEnum):
    _pass = 0
    _bet = 1


class Players(enum.IntEnum):
    one = 0
    two = 1


class Stage(enum.IntEnum):
    one = 0
    two = 1
    final = 2


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

    def t(self):
        pass

    def r(self):
        pass

    def step(self, action: int):
        """
        :param action: Integer representation of the action to execute in the environment.
        :return: state, reward, done, info
        """

