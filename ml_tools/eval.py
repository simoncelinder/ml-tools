import pandas as pd
import datetime


def cv_time_series(
        df: pd.DataFrame,
        model_ref: any,
        feature_list: list,
        label: str = 'label',
        hypers: dict = {},
        pred: str = 'cv_pred',
        cv_start: datetime.date = datetime.date(2021, 6, 10),
        step_days=10
) -> pd.DataFrame:

    """ Expanding Window Cross-validation for time series """

    df = df.copy()
    split = cv_start
    model = model_ref(**hypers)

    if feature_list is None:
        feature_list = [i for i in df.columns if i != label]

    while split < df.index.date.max():
        cv_train = df[:split]
        cv_test = df.loc[df.index > cv_train.index.max()]
        model.fit(
            cv_train[feature_list],
            cv_train[label],
        )
        df.loc[df.index.isin(cv_test.index), pred] = (
            model.predict(cv_test[feature_list])
        )
        split += datetime.timedelta(days=step_days)

    return df


def get_mae_from_cv_time_series(
        df: pd.DataFrame,
        model_ref: any,
        feature_list: list = None,
        cv_start: datetime.date = datetime.date(2021, 6, 10),
        label: str = 'label',
        hypers: dict = {},
        pred: str = 'cv_pred',
        step_days: int = 10,
) -> float:

    """ Wrapper for just getting MAE score from cv for time series """

    df = cv_time_series(
        df=df,
        feature_list=feature_list,
        model_ref=model_ref,
        cv_start=cv_start,
        label=label,
        hypers=hypers,
        pred='cv_pred',
        step_days=step_days
    )

    return (df[label] - df[pred]).abs().mean()
