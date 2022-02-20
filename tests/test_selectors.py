import datetime
import numpy as np
from lightgbm import LGBMRegressor

from ml_tools.datasets import generate_synthetic_data
from ml_tools.feature_selector import FeatureSelector
from ml_tools.eval import get_mae_from_cv_time_series
from ml_tools.train_start_selector import TrainStartSelector


def test_feature_selector_discards_noise():

    # Dataset that can be learned from just day of year and day of week
    df = generate_synthetic_data(
        freq='d',
        weekday_offset=True,
        yearly_offset=True,
        start_value=100,
        trend_stop_value=100
    )

    noise = np.random.normal(size=len(df), scale=20)
    df['signal_with_noise'] = df.label + noise
    df['dow'] = df.index.weekday
    df['doy'] = df.index.dayofyear

    fs = FeatureSelector()
    best_features = fs.run(
        eval_func=get_mae_from_cv_time_series,
        df=df,
        model=LGBMRegressor(),
        feature_list=[i for i in df.columns if i != 'label']
    )

    assert set(best_features) == {'doy', 'dow'}


def test_feature_selector_selects_improving_signal():

    # With only one signal that improves, and one that decays,
    # should learn to listen to improving signal given recent cv window
    df = generate_synthetic_data(
        freq='d',
        weekday_offset=True,
        yearly_offset=True,
        start_value=100,
        trend_stop_value=100
    )
    noise = np.random.normal(size=len(df), scale=0.3)
    df['signal_decay'] = df.label + noise * np.linspace(30, 100, len(df))
    df['signal_improves'] = df.label + noise * np.linspace(100, 30, len(df))

    fs = FeatureSelector()
    best_features = fs.run(
        eval_func=get_mae_from_cv_time_series,
        df=df,
        model=LGBMRegressor(),
        feature_list=[i for i in df.columns if i != 'label']
    )

    assert set(best_features) == {'signal_improves'}


def test_train_selector_drops_very_noisy_start():

    df = generate_synthetic_data(
        freq='d',
        weekday_offset=True,
        yearly_offset=True,
        trend_stop_value=100
    )

    reasonable_noise = np.random.normal(size=len(df), scale=5)
    df['feature'] = df.label + reasonable_noise

    chaos_slice = df.index < '2020-01-01'
    df.loc[chaos_slice, 'feature'] = (
            df.loc[chaos_slice, 'label'] +
            np.random.normal(size=sum(chaos_slice), scale=50)
    )

    cv_start = datetime.date(2021, 6, 10)

    tss = TrainStartSelector(
        eval_window_rows=len(df[cv_start:]),
        min_train_rows=7 * 4,
        n_trials=20
    )

    best_train_start = tss.run(
        eval_func=get_mae_from_cv_time_series,
        df=df,
        model=LGBMRegressor(),
        cv_start=cv_start
    )

    assert best_train_start >= df[chaos_slice].index.max()
    assert best_train_start < cv_start
