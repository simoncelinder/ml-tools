import numpy as np
import pandas as pd


def get_yearly_sin_seasonality(
        df: pd.DataFrame,
        normalize_factor: float = 0.2
) -> pd.Series:
    def get_sin(row):
        doy = row.name.dayofyear
        return np.sin(2 * np.pi * doy / 366) * normalize_factor + 1

    return df.apply(get_sin, axis=1)


def generate_synthetic_data(
        hour_offset: bool = False,
        weekday_offset: bool = False,
        yearly_offset: bool = False,
        freq: str = 'h',
        start: str = '2019-08-01 00:00',
        stop: str = '2022-01-07 23:00',
        start_value: float = 100,
        trend_stop_value: float = 400,
        label: str = 'label',
) -> pd.DataFrame:

    # Create placeholder df
    df = pd.DataFrame(index=[pd.to_datetime(start), pd.to_datetime(stop)])
    df = df.asfreq(freq)

    # Create linear trend
    if trend_stop_value is not None:
        df[label] = np.linspace(start_value, trend_stop_value, len(df))
    else:
        df[label] = start_value

    if yearly_offset:
        df[label] *= get_yearly_sin_seasonality(df)

    # Offset hour of day
    if hour_offset:
        assert freq == 'h'
        hour_offset_dict = {
            0: 0.2,
            1: 0.1,
            2: 0.1,
            3: 0.05,
            4: 0.05,
            5: 0.1,
            6: 0.5,
            7: 0.7,
            8: 1.1,
            9: 1.2,
            10: 1.1,
            11: 1,
            12: 1,
            13: 0.9,
            14: 1,
            15: 1.1,
            16: 1.15,
            17: 1.1,
            18: 1.1,
            19: 0.9,
            20: 0.85,
            21: 0.75,
            22: 0.5,
            23: 0.3
        }
        for hour in hour_offset_dict.keys():
            df.loc[df.index.hour == hour, label] = (
                    df.loc[df.index.hour == hour, label] * hour_offset_dict[hour])

    # Offset day of week
    if weekday_offset:
        weekday_offset_dict = {
            0: 1.2,
            1: 1.1,
            2: 1.05,
            3: 1,
            4: 1,
            5: 0.90,
            6: 0.90
        }
        for weekday in weekday_offset_dict.keys():
            df.loc[df.index.weekday == weekday, label] = (
                    df.loc[df.index.weekday == weekday, label] * weekday_offset_dict[weekday])

    return df
