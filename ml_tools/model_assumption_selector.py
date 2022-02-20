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
            selectors: tuple  # E.g: (TrainStartSelector(...), FeatureSelector(...), Tuner(...))
    ):

        self.selectors = selectors

    def run(
            self,
            eval_func,
            **kwargs
    ):

        best_assumptions = {}
        for selector in self.selectors:
            best_assumptions[selector.__class__.__name__] = (
                selector.run(eval_func, **kwargs)
            )

        return best_assumptions
