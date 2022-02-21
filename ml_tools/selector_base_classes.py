class TrainStartSelectorBase:
    @staticmethod
    def update_kwargs(kwargs, search_result, verbosity):
        kwargs['df'] = kwargs['df'].loc[kwargs['df'].index >= search_result]
        if verbosity >= 1:
            print(f"Updated df to selected train start: {search_result}")


class FeatureSelectorBase:
    @staticmethod
    def update_kwargs(kwargs, search_result, verbosity):
        kwargs['feature_list'] = search_result
        if verbosity >= 1:
            print(f"Updated feature_list to selected: {search_result}")


class TunerBase:
    @staticmethod
    def update_kwargs(kwargs, search_result, verbosity):
        kwargs['hypers'] = search_result
        if verbosity >= 1:
            print(f"Updated hyperparameters to selected: {search_result}")
