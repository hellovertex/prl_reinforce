from typing import Optional, Union, Any, Dict

import numpy as np
from tianshou.data import Batch
from tianshou.policy import RainbowPolicy


class Rainbow(RainbowPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._card_stength_model = None  # set lazily, so we don't have to change all dependent code
        self._mode_append_winprob = False

    @property
    def card_stength_model(self):
        return self._card_stength_model

    @card_stength_model.setter
    def card_stength_model(self, val):
        self._card_stength_model = val

    @property
    def mode_append_winprob(self):
        return

    @mode_append_winprob.setter
    def mode_append_winprob(self, value: bool):
        self._mode_append_winprob = value

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            model: str = "model",
            input: str = "obs",
            **kwargs: Any,
    ) -> Batch:
        if self._mode_append_winprob:
            # todo: check concatenation
            win_prob = self._card_stength_model(batch.obs.obs)
            batch.obs.obs = np.concatenate([batch.obs.obs, win_prob])
            # todo: check repeat for obs_next ?
            if hasattr(batch, 'obs_next'):
                win_prob = self._card_stength_model(batch.obs.obs)
                batch.obs_next.obs = np.concatenate([batch.obs.obs, win_prob])
        return super().forward(batch=batch, state=state, model=model, input=input)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._mode_append_winprob:
            # todo: check concatenation
            win_prob = self._card_stength_model(batch.obs.obs)
            batch.obs.obs = np.concatenate([batch.obs.obs, win_prob])
            # todo: check repeat for obs_next ?
            if hasattr(batch, 'obs_next'):
                win_prob = self._card_stength_model(batch.obs.obs)
                batch.obs_next.obs = np.concatenate([batch.obs.obs, win_prob])
        return super().learn(batch)
