from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class FeatureSelector:

    """ This class is used to run a feature selection heuristic on a holdout set or with cross-validation. If the
    user has ranked the feature_list so features assumed to have higher explanatory power are earlier in the list,
    the heuristic can then recursively try to add (from start of list) or remove (from end of list) one feature
    at a time, up to a specific search depth, to see which feature to added/removed would improve the score the most
    within that search depth of the list (given that they might interact). The user sends in what evaluation function
    to use, and whether the result of this function should be maximized or minimized. The user can also set patience
    (e.g. if many features), so if the search doesnt improve over a certain number of iterations of adding/removing
    features, the search stops and the best feature_list so far is returned.
    If feature_list is not sorted, the class can still be used to search through what features would
    improve vs not improve the score and return the best feature list by setting infinite patience to search
    through all of them. """

    direction: str = 'minimize'  # Whether to maximize or minimize the metric
    strategy: str = 'adding'  # Whether to recursively add or remove features
    search_depth: int = 4  # How many features to test adding/removing one at a time during each loop
    patience: int = 2  # How many iterations to run without improvement without the search stopping
    add_feature_threshold: float = 0.0001  # When adding features, require small improvement in score

    verbosity: float = 1
    result_df: pd.DataFrame = None

    def run(
            self,
            eval_func,  # Should get a score to minimize or maximize, e.g. from cross-validation
            **kwargs,  # Set all arguments to eval_func when falling
    ):
        self.assert_consistency(eval_func, kwargs)
        list_of_dicts = []

        # Get baseline score for all features, no selection
        global_best_score = self.run_baseline_all_features(eval_func, kwargs, list_of_dicts)
        best_feature_list = kwargs['feature_list']

        # Init other variables before loop
        patience_counter, remaining_features_list, selected_features_list = (
            self.init_variables(feature_list=kwargs['feature_list']))

        # Use strict > 1 here to avoid double calculating scenario of all features
        while len(remaining_features_list) > 1:
            best_feature, best_score, global_improvement = self.reset_variables_current_trial()

            # Gets beginning of list if adding, end of list if removing
            search_feature_list = self.get_search_feature_list(remaining_features_list)

            for feature in search_feature_list:
                if self.verbosity >= 2:
                    print(f"{self.strategy} feature {feature}")

                # Update feature combination based on strategy (removing vs adding the feature)
                feature_combination_list = self.update_current_feature_combination(feature, selected_features_list)

                # Add current feature combination to input kwargs to scoring function
                kwargs['feature_list'] = feature_combination_list

                # Get score for current iteration
                current_score = eval_func(**kwargs)

                # Update local variables if improvement this iteration
                best_feature, best_score = self.update_local_variables(current_score, best_score, feature, best_feature)

                # Update global variables if global improvement of score
                best_feature_list, global_best_score, global_improvement = self.update_global_variables(
                    current_score, feature_combination_list, global_best_score, global_improvement, feature,
                    best_feature_list)

            # Check patience vs global improvement (if no improvement for too many rounds, stop search)
            if global_improvement:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > self.patience:
                    break

            # Update feature lists
            self.update_feature_lists(best_feature, remaining_features_list, selected_features_list)

            if self.verbosity >= 1.5:
                print('\n', f"Finalizing iteration: Best score: {round(best_score, 4)} "
                      f"with features {selected_features_list}")

            # Append best score for current number of features
            list_of_dicts.append({'num_features': len(feature_combination_list),
                                  'score': best_score})

        # Print out final result of search
        if self.verbosity >= 0.5:
            print('\n', f"FeatureSelector - Final best score: {round(global_best_score, 3)} "
                        f"with features: {best_feature_list}")

        # Summarize result in dataframe as attribute
        self.result_df = pd.DataFrame(list_of_dicts).set_index('num_features')
        self.result_df = self.result_df.sort_index(ascending=True)

        return best_feature_list

    def assert_consistency(self, eval_func, kwargs):
        # Interface requirement: eval_func must take feature_list as an argument for loop to work,
        # feature_list which is full list to search, needs to be in kwargs, but is recursively updated in the loop
        assert 'feature_list' in eval_func.__code__.co_varnames
        assert 'feature_list' in kwargs.keys()
        assert self.direction in ['minimize', 'maximize']
        assert self.strategy in ['adding', 'removing']

    def run_baseline_all_features(self, eval_func, kwargs, list_of_dicts):
        current_score = eval_func(**kwargs)
        if self.verbosity >= 1:
            print("Baseline score with all features", round(current_score, 4))
        list_of_dicts.append({'num_features': len(kwargs['feature_list']),
                              'score': current_score})
        global_best_score = current_score
        return global_best_score

    def update_feature_lists(self, best_feature, remaining_features_list, selected_features_list):
        if self.strategy == 'adding':
            remaining_features_list.remove(best_feature)
            selected_features_list.append(best_feature)
        else:
            # Remove strategy - removes from both lists
            remaining_features_list.remove(best_feature)
            selected_features_list.remove(best_feature)

    def update_current_feature_combination(self, feature, selected_features_list):
        if self.strategy == 'adding':
            feature_combination_list = selected_features_list + [feature]
        else:
            # Remove the current feature from the list
            feature_combination_list = selected_features_list.copy()
            feature_combination_list.remove(feature)
        return feature_combination_list

    def get_search_feature_list(self, remaining_features_list):
        if self.strategy == 'adding':
            # Adding from beginning of list
            search_feature_list = remaining_features_list[0:min(self.search_depth, len(remaining_features_list))]
        else:
            # Removing from end of list, reverse order of list
            search_feature_list = remaining_features_list[-min(self.search_depth, len(remaining_features_list)):][::-1]
        return search_feature_list

    def update_global_variables(self, current_score, feature_combination_list, global_best_score, global_improvement,
                                feature, best_feature_list):

        if self.strategy == 'removing':
            threshold = 0  # Overrule with Occams Razor for removing features, fewer features better if same score
        else:
            threshold = self.add_feature_threshold

        if (self.direction == 'minimize' and current_score <= global_best_score + threshold) or \
                (self.direction == 'maximize' and current_score > global_best_score + threshold):

            global_best_score = current_score
            best_feature_list = feature_combination_list
            global_improvement = True

            if global_improvement and self.verbosity >= 1:
                print('\n', f"* * Global loss improved: {round(current_score, 5)} with {len(feature_combination_list)} "
                            f"features, feature_list = {feature_combination_list} "
                      f"{self.strategy} feature {feature} * *")

        return best_feature_list, global_best_score, global_improvement

    def reset_variables_current_trial(self):
        best_feature = None
        if self.direction == 'maximize':
            best_score = 0
        else:
            best_score = np.inf
        global_improvement = False  # Reset
        return best_feature, best_score, global_improvement

    def update_local_variables(self, current_score, best_score, feature, best_feature):
        if current_score != best_score:
            if self.direction == 'minimize' and current_score < best_score:
                best_score = current_score
                best_feature = feature
            elif self.direction == 'maximize' and current_score > best_score:
                best_score = current_score
                best_feature = feature
        return best_feature, best_score

    def plot_result(self):
        assert self.result_df is not None, 'No result yet to plot'
        self.result_df.plot(), plt.show()

    def init_variables(self, feature_list):
        patience_counter = 0
        if self.strategy == 'adding':
            selected_features_list = []
        else:
            # If remove strategy, add all features to list to be recursively removed from
            selected_features_list = feature_list.copy()
        remaining_features_list = feature_list.copy()
        return patience_counter, remaining_features_list, selected_features_list
