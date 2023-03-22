from typing import Optional, Union, Any, Dict

import numpy as np
from tianshou.data import Batch
from tianshou.policy import RainbowPolicy


class Rainbow(RainbowPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mc_model = None  # set lazily, so we don't have to change all dependent code
        self._mode_append_winprob = False

    @property
    def mc_model(self):
        return self._mc_model

    @mc_model.setter
    def mc_model(self, val):
        self._mc_model = val

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
            # todo: overwrite batch with winprob and call super
            pass
        return super().forward(batch=batch, state=state, model=model, input=input)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._mode_append_winprob:
            # todo: overwrite batch with winprob and call super
            pass
        return super().learn(batch)
