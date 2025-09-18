import abc
import time
import typing
import os
from typing import Any

import pandas as pd
from utils.storage import save
from algorithms.generic_algorithm import GenericAlgorithm
from algorithms.causal_order.causal_order_result import CausalOrderResult

CausalOrder: typing.TypeAlias = list[int]


class GenericCausalOrderAlgorithm(GenericAlgorithm):
    """
    An abstract base class representing a generic algorithm that produces a causal order
    """
    @abc.abstractmethod
    def run(self, df: pd.DataFrame) -> CausalOrder:
        """
        Run the Causal Order Algorithm

        Parameters
        ----------
        df : pd.DataFrame,
             The training IT_monitoring, with shape (n_samples, n_features) where
             - n_samples is the number of samples
             - n_features is the number of features

        Returns
        ----------
        causal_order : CausalOrder,
            The causal order. A CausalOrder is a list[int] where
            each integer represents the index of a feature in the training IT_monitoring.
        """
        raise NotImplementedError

    def _algorithm_type(self) -> str:
        return "Causal Order"

    def _algorithm_dir(self) -> str:
        return "causal_order"

    def format_result(self, causal_order: CausalOrder, time_taken: float) -> CausalOrderResult:
        return CausalOrderResult(causal_order, time_taken, self._algorithm_type(), self.__str__(), self.target_file)

    def get_causal_order_result(self, filepath: str) -> CausalOrderResult:
        """
        Run the Causal Order Algorithm

        Parameters
        ----------
        filepath : str
            The path to the file to load the training IT_monitoring from.
            Only .csv, .xls, and .xlsx files are currently supported.

        Returns
        ----------
        causalOrderResult : CausalOrderResult
            An object containing:
            - `causal_order`: list of feature indices representing the causal order.
            - `time_taken`: time taken to compute the causal order, in seconds.
        """
        return self._get_result(filepath)

    def get_and_save_causal_order_result(self, filepath: str) -> CausalOrderResult:
        """
        Run the Causal Order Algorithm and pickle the result.

        Parameters
        ----------
        filepath : str
            The path to the file to load the training IT_monitoring from.
            Only .csv, .xls, and .xlsx files are currently supported.

        Returns
        ----------
        causalOrderResult : CausalOrderResult
            An object containing:
            - `causal_order`: list of feature indices representing the causal order.
            - `time_taken`: time taken to compute the causal order, in seconds.
        """
        return self._get_and_save_result(filepath)
