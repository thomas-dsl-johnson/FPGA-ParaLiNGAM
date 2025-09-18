from abc import ABC
from typing import Any

from algorithms.end_to_end.end_to_end_result import EndToEndResult
from algorithms.end_to_end.generic_end_to_end_algorithm import GenericEndToEndAlgorithm
import lingam

class DirectLingamEndToEndAlgorithm(GenericEndToEndAlgorithm, ABC):
    """
    Runs the full DirectLiNGAM algorithm using the LiNGAM module.
    """
    def format_result(self, model: lingam.DirectLiNGAM, time_taken: float) -> EndToEndResult:
        return EndToEndResult.from_model(model, time_taken, self._algorithm_type(), self.__str__(), self.target_file)

    def run(self, df) -> Any:
        """
        Run the lingam module's DirectLiNGAM algorithm

        Parameters
        ----------
        df : pd.DataFrame,
             The training data, with shape (n_samples, n_features) where
             - n_samples is the number of samples
             - n_features is the number of features

        Returns
        ----------
        mode : lingam.DirectLiNGAM,
            The model, encapsulating the result of the DirectLiNGAM algorithm.
            See https://github.com/cdt15/lingam/blob/master/lingam/direct_lingam.py for more information.
        """
        model = lingam.DirectLiNGAM()
        model.fit(df)
        return model

    def __str__(self):
        return "DirectLingamEndToEndAlgorithm"




