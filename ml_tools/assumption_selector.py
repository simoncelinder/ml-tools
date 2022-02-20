from dataclasses import dataclass


@dataclass
class FeatureSelector:
    # Already impelmented but maybe extend with some more search strategies (binary search of how many features when sorted)
    def run(self, eval_func, **kwargs):
        pass


@dataclass
class TrainStartSelector:
    # Binary search or more open search of optimal train start
    def run(self, eval_func, **kwargs):
        pass


@dataclass
class Tuner:
    # Wrapper for Optuna for ML
    def run(self, eval_func, **kwargs):
        pass


@dataclass
class AssumptionSelector:

    """ Selection of best assumptions for an ML model"""

    selectors: tuple = (TrainStartSelector(), FeatureSelector(), Tuner())

    def get_best_model_assumptions(
            self,
            eval_func,
            **kwargs
    ):

        best_assumptions = []
        for selector in self.selectors:
            best_assumptions.append(selector.run(eval_func, **kwargs))

        return best_assumptions



    """  OLD
    def get_best_model_and_assumpsions(self, model_ref, eval_func, default_hypers=None, tune_ranges=None, **kwargs):
        best_train_start = None if self.tss is None else self.tss.get_best_train_start(eval_func, **kwargs)
        best_features = None if self.fs is None else self.fs.get_best_features(eval_func, **kwargs)
        best_hypers = None if self.tuner is None else self.tuner.get_best_hypers(eval_func, tune_ranges, **kwargs)
        model = model_ref(**default_hypers) if self.tuner is None else model_ref(**best_hypers)
        return model, best_train_start, best_features, best_hypers
        """
