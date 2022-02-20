from ml_tools.feature_selector import FeatureSelector
from ml_tools.train_start_selector import TrainStartSelector
from ml_tools.tuner import Tuner


class ModelAssumptionSelector:

    """ Select of best assumptions for an ML model. Idea is to have an interface that is agnostic to
    what evaluation function you'd want to use (cross-val, validation set), and in what order you would
    like to search within each concept. (While some suggest it's best to search simultaneously in the full
    hyperparam space (space of assumptions), there might be advantages to exploring concepts sequentially,
    since the search space becomes more manageable and the optimizers can focus on one concept at a time.)

    Example call:

    mas = ModelAssumptionSelector(
        selectors = (

            TrainStartSelector(
                eval_window_rows = len(df[cv_start:]),
                min_train_rows= 7*4,
                n_trials=10
            ),

            FeatureSelector(),

            Tuner(params_grid={'dummy': 1})  # TODO

        )
    )

    best_assumptions = mas.run(
        eval_func=get_mae_from_cv_time_series,
        df=df,
        model=lgb.LGBMRegressor(),  # TODO pass by reference for tuning, model = model_ref(**params)
        feature_list=[i for i in df.columns if i != 'label']
    )

     """

    def __init__(
            self,
            selectors: tuple,  # E.g: (TrainStartSelector(...), FeatureSelector(...), Tuner(...))
            verbosity: int = 1,
    ):

        self.selectors = selectors
        self.verbosity = verbosity


    def run(
            self,
            eval_func,
            **kwargs
    ):

        best_assumptions = {}
        for selector in self.selectors:

            search_result = selector.run(eval_func, **kwargs)
            best_assumptions[selector.__class__.__name__] = search_result

            if isinstance(selector, TrainStartSelector):
                self.update_data_start(kwargs, search_result)

            # Might want to check why these 2 required .__name__
            # matching and isinstance didn't match
            elif selector.__class__.__name__ == FeatureSelector.__name__:
                self.update_feature_list(kwargs, search_result)

            elif selector.__class__.__name__ == Tuner.__name__:
                self.update_hypers(kwargs, search_result)

        return best_assumptions

    def update_data_start(self, kwargs, search_result):
        kwargs['df'] = kwargs['df'].loc[kwargs['df'].index >= search_result]
        if self.verbosity >= 1:
            print(f"Updated df to selected train start: {search_result}")

    def update_feature_list(self, kwargs, search_result):
        kwargs['feature_list'] = search_result
        if self.verbosity >= 1:
            print(f"Updated feature_list to selected: {search_result}")

    def update_hypers(self, kwargs, search_result):
        kwargs['hypers'] = search_result
        if self.verbosity >= 1:
            print(f"Updated hyperparameters to selected: {search_result}")
