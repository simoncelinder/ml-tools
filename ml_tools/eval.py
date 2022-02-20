import pandas as pd
import datetime


def cv_time_series(
        df: pd.DataFrame,
        model: any,
        features: list,
        label: str = 'label',
        pred: str = 'cv_pred',
        cv_start: datetime.date = datetime.date(2021, 6, 10),
        step_days=10
) -> pd.DataFrame:

    """ Expanding Window Cross-validation for time series """

    df = df.copy()
    split = cv_start

    if features is None:
        features = [i for i in df.columns if i != label]

    while split < df.index.date.max():
        cv_train = df[:split]
        cv_test = df.loc[df.index > cv_train.index.max()]
        model.fit(
            cv_train[features],
            cv_train[label],
        )
        df.loc[df.index.isin(cv_test.index), pred] = (
            model.predict(cv_test[features])
        )
        split += datetime.timedelta(days=step_days)

    return df


def get_mae_from_cv_time_series(
        df: pd.DataFrame,
        model: any,
        feature_list: list = None,
        cv_start: datetime.date = datetime.date(2021, 6, 10),
        label: str = 'label',
        pred: str = 'cv_pred',
        step_days: int = 10,
) -> float:

    """ Wrapper for just getting MAE score from cv for time series """

    df = cv_time_series(
        df=df,
        features=feature_list,
        model=model,
        cv_start=cv_start,
        label=label,
        pred='cv_pred',
        step_days=step_days
    )

    return (df[label] - df[pred]).abs().mean()
