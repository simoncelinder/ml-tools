import optuna
from ml_tools.selector_base_classes import TrainStartSelectorBase
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TrainStartSelector(TrainStartSelectorBase):

    """ To select where good signal starts in the data, since old data might not be so relevant any more.
    This can be relevand not only for time series forecasting and regression, but also classification cases
    where old data might no longer be very valid. Suggested to explore with reasonable cross-validation window.
    Assumes input dataframe is sorted so older data appears on top. """

    def __init__(
            self,
            eval_window_rows: int,  # Exclude eval window from slicing for min_train_rows
            min_train_rows: int = 1000,
            direction: str = 'minimize',
            n_trials: int = 30
    ):
        assert direction in ['minimize', 'maximize']
        self.min_train_rows = min_train_rows
        self.eval_window_rows = eval_window_rows
        self.direction = direction
        self.n_trials = n_trials
        self.original_df = None

    def run(
            self,
            eval_func,  # Should get a score to minimize or maximize, e.g. from cross-validation
            **kwargs,  # Set all arguments to eval_func when falling
    ):
        assert 'df' in kwargs.keys(), 'df must be in kwargs'
        self.original_df = kwargs['df'].copy()

        # Note: Tried more light-weight implementation with scipy-optimize but ran into troubles
        # (Not fully continuous loss function)
        def objective(trial):
            iloc_start = trial.suggest_int(
                'iloc_start',
                0,  # Min iloc to include even oldest data
                len(kwargs['df']) - self.eval_window_rows - self.min_train_rows,  # Max iloc matching
            )
            kwargs['df'] = self.original_df.iloc[int(iloc_start):]
            return eval_func(**kwargs)

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials)
        best_train_start_idx = self.original_df.iloc[study.best_params['iloc_start']:].index.min()

        return best_train_start_idx
