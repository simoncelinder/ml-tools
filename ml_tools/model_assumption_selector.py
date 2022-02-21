class ModelAssumptionSelector:

    """ Select of best assumptions for an ML model. Idea is to have an interface that is agnostic to
    what evaluation function you'd want to use (cross-val, validation set), and in what order you would
    like to search within each concept. (While some suggest it's best to search simultaneously in the full
    hyperparam space (space of assumptions), there might be advantages to exploring concepts sequentially,
    since the search space becomes more manageable and the optimizers can focus on one concept at a time.) """

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
            selector.update_kwargs(kwargs, search_result, self.verbosity)

        return best_assumptions
