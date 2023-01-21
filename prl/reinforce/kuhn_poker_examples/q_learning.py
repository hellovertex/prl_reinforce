import random
from q_value_iteration import S, N_ACTIONS, N_STATES


class KuhnPokerEnvironment:
    """This class provides an interface for an agent to interact with
    the underlying MDP of Kuhn Poker"""

    def __init__(self):
        self._state = None
        self._deck = ['J', 'Q', 'K']
        self._player_hands = {'Player one': None,
                              'Player two': None}
        self.actions_taken = []
        self.stage = 0

    def draw_card(self):
        assert len(self._deck) > 0
        card = random.choice(self._deck)
        index = self._deck.index(card)
        self._deck.remove(card)
        return card, index

    def reset(self):
        # Draw card for Player one
        cname, cid = self.draw_card()
        self._player_hands['Player one'] = cname

        # Draw card for Player two
        cname, _ = self.draw_card()
        self._player_hands['Player two'] = cname

        state = S(cid)
        return state 

    def step(self, action: int):
        """
        :param action: Integer representation of the action to execute in the environment.
        :return: state, reward, done, info
        """
        # state 0,1,2:
        # check or bet
