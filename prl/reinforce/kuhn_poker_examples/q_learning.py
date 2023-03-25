from copy import deepcopy
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

