import abc
import os
import time
from typing import Any
import pandas as pd
from utils.storage import save


class GenericAlgorithm:
    """
    An abstract base class representing a generic algorithm that produces a causal order and model
    """
    def __init__(self):
        self.target_file = ""


    @abc.abstractmethod
    def run(self, df: pd.DataFrame) -> Any:
        """
        Run the Algorithm

        Parameters
        ----------
        df : pd.DataFrame,
             The training data, with shape (n_samples, n_features) where
             - n_samples is the number of samples
             - n_features is the number of features

        Returns
        ----------
        result : Any
            The result of the algorithm
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        The default string representation of the algorithm.
        Used in the name of the .pkl when saving the model.

        Returns
        ----------
        string : str
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _algorithm_type(self) -> str:
        """
        The type of the algorithm i.e. is the algorithm a causal order algorithm or an end-to-end algorithm

        Returns
        ----------
        string : str
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _algorithm_dir(self) -> str:
        """
        The dir of the algorithm i.e. Where to save results of the algorithm within the results folder

        Returns
        ----------
        string : str
        """
        raise NotImplementedError

    @abc.abstractmethod
    def format_result(self, result: Any, time_taken: float) -> Any:
        """
        Formats the result and the time taken to be returned by get_result()

        Parameters
        ----------
        result : Any
            The result of running the algorithm.
        time_taken : float
            The time taken to run the algorithm.

        Returns
        ----------
        formatted_result : Any
        """
        raise NotImplementedError

    @staticmethod
    def __get_df(filepath: str) -> pd.DataFrame:
        file_extension = os.path.splitext(filepath)[1].lower()
        if file_extension == '.csv':
            X = pd.read_csv(filepath)
        elif file_extension in ['.xls', '.xlsx']:
            X = pd.read_excel(filepath)
        else:
            raise ValueError(
                f"Unsupported file type: '{file_extension}'. "
                "Only .csv, .xls, and .xlsx files are currently supported."
            )
        # If the first column is dates, remove it
        first_column = X.columns[0]
        if pd.api.types.is_datetime64_any_dtype(X[first_column]) or pd.to_datetime(X[first_column],
                                                                                   errors='coerce').notna().all():
            X = X.iloc[:, 1:]

        return X

    def _get_result(self, filepath: str) -> Any:
        """
        Run the Algorithm and add time taken to the result

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
        self.target_file = filepath
        df = self.__get_df(filepath)
        beg = time.time()
        result = self.run(df)
        end = time.time()
        time_taken = end - beg
        return self.format_result(result, time_taken)

    def _get_save_location(self, filepath: str) -> str:
        file_name = os.path.splitext(filepath)[0].lower()
        return self._algorithm_dir() + "/" + self.__str__() + "/" + os.path.basename(os.path.dirname(os.path.dirname(filepath))) + "/" + os.path.basename(os.path.dirname(filepath)) + "/"+ os.path.basename(file_name) + ".pkl"


    def _get_and_save_result(self, filepath: str) -> Any:
        """
        Run the Causal Order Algorithm and pickle the result.

        Parameters
        ----------
        filepath : str
            The path to the file to load the training IT_monitoring from.
            Only .csv, .xls, and .xlsx files are currently supported.

        Returns
        ----------
        result : EndToEndResult
            An object that encapsulates the estimated causal order and the fitted DirectLiNGAM model.
            - `result.causal_order_result.causal_order` gives the causal order as a list of feature indices.
            - `result.causal_order_result.time` gives the time taken in seconds.
            - `result.model` gives access to the full fitted lingam.DirectLiNGAM model.
        """
        result = self._get_result(filepath)
        result_location = self._get_save_location(filepath)
        save(result, result_location)
        return result