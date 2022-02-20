import optuna


class Tuner:

    def __init__(
            self,
            params_grid: dict,
            direction: str = 'minimize',
    ):
        self.direction = direction
        self.params_grid = params_grid

    def run(
            self,
            eval_func,  # Should get a score to minimize or maximize, e.g. from cross-validation
            **kwargs,  # Set all arguments to eval_func when falling
    ):

        not_implemented = True
        if not_implemented:
            print("Tuner not implemented yet!")

        else:

            def objective(trial):
                # TODO
                return eval_func(**kwargs)

            study = optuna.create_study(direction=self.direction)
            study.optimize(objective, n_trials=self.n_trials)
            return study.best_params
