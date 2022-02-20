import optuna
import functools


class Tuner:

    def __init__(
            self,
            lazy_optuna_space: functools.partial,  # e.g. partial({'learnin_rate': trial.suggest_ ..., ...})
            direction: str = 'minimize',
            n_trials: int = 40,
            verbosity: int = 1
    ):
        self.direction = direction
        self.lazy_optuna_space = lazy_optuna_space
        self.n_trials = n_trials
        self.verbosity = verbosity

    def run(
            self,
            eval_func,  # Should get a score to minimize or maximize, e.g. from cross-validation
            **kwargs,  # Set all arguments to eval_func when falling
    ):

        # Get baseline score with out-of-box hyperparameters of model
        kwargs['hypers'] = {}
        baseline_score = eval_func(**kwargs)

        if self.verbosity >= 1:
            print(f"Baseline score with out-of-box hyperparamers {baseline_score :.2f}")

        def objective(trial):

            # Call evaluation of optuna suggestions on the trial
            kwargs['hypers'] = self.lazy_optuna_space(trial)
            score = eval_func(**kwargs)
            if self.verbosity >= 1:
                print(f"""Trial {trial.number}, got result {score :.2f} with hypers {kwargs['hypers']}""")
            return score
            
        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials)

        best_params = self.compare_to_baseline(baseline_score, study)

        return best_params

    def compare_to_baseline(self, baseline_score, study):
        if baseline_score <= study.best_value:
            if self.direction == 'minimize':
                best_params = {}
                if self.verbosity >= 1:
                    print("Did not beat out-of-box hyperparameters during tuning, using them instead")
            else:
                best_params = study.best_params
        else:
            if self.direction == 'minimize':
                best_params = study.best_params
            else:
                best_params = {}
        return best_params
