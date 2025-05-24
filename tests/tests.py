import numpy as np
import pandas as pd
import unittest
import functools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from matplotlib.testing.compare import compare_images

from heatmapts import heatmapfigure, HeatmapFigure
from heatmapts.suntimes import Sun


PATH_TEST_OUTPUT = "tests/result_images"
PATH_BASELINE_IMAGES = "tests/baseline_images"


def compare_to_baseline(func, tol=1, file_format=".png", dpi=500):
    """Custom decorator to compare returned figure to baseline saved on disk.
    Could not get the one included in matplotlib working with unittest.
    Helpful info also here:
    https://www.davidketcheson.info/2015/01/13/using_matplotlib_image_comparison.html
    """
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        filename = func.__name__ + ".png"

        out = func(*args, **kwargs)
        if out is not None:
            out.savefig(f'{PATH_TEST_OUTPUT}/{filename}', dpi=dpi)

        result = compare_images(
            f'{PATH_BASELINE_IMAGES}/{filename}',
            f'{PATH_TEST_OUTPUT}/{filename}',
            tol=tol,
        )
        if result is not None:
            raise AssertionError(f"Image comparison failed: {result}")
    return decorated


def sinus_pattern(timestamp, binary=True):
    first_hour = 2 * np.sin((timestamp.dayofyear + 107) / 365 * 2 * np.pi) + 5.25
    last_hour = -2 * np.sin((timestamp.dayofyear + 107) / 365 * 2 * np.pi) + 17.5
    hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600
    if binary:
        ret = 1 if (hour > first_hour) and (hour < last_hour) else 0
    else:
        ret = 1 / (1 + np.exp(-((hour - first_hour) * (last_hour - hour) / 10)))
    return ret


class TestHeatmap(unittest.TestCase):
    @compare_to_baseline
    def test_histogram_alignment(self):
        series = pd.Series(
            index=pd.date_range('2020-01-01', '2020-01-02 23:59:59', freq='15min', tz='UTC'),
            data=[1, 0, 1] + 90 * [0] + [1, 0, 2] + [0] * 96
        )
        return heatmapfigure(series, rasterized=True)

    @compare_to_baseline
    def test_negative_peaks(self):
        series = pd.Series(
            index=pd.date_range('2020-01-01', '2020-01-02 23:59:59', freq='15min', tz='UTC'),
            data=[1, 0, -1] + 90 * [0] + [-1, 0, -2] + [0] * 96
        )
        return heatmapfigure(series, rasterized=True)

    @compare_to_baseline
    def test_suntimes(self):
        date_range = pd.date_range('2020-01-01', '2021-01-01', freq='15min', tz='UTC')
        series = pd.Series(index=date_range, data=[0, 1] * int((len(date_range) / 2)) + [0])
        lat, long = 47.492, 8.555
        return heatmapfigure(series, rasterized=True, annotate_suntimes=(lat, long))

    @compare_to_baseline
    def test_na(self):
        series = pd.Series(
            index=pd.date_range('2020-01-01', '2020-01-02 23:59:59', freq='15min', tz='UTC'),
            data=[1, 0, 1] + 90 * [0] + [1, 0, 2] + [0] * 96
        )
        series.iloc[10:20] = pd.NA
        return heatmapfigure(series, rasterized=True)

    @compare_to_baseline
    def test_time_change(self):
        date_range = pd.date_range('2021-10-30', '2021-11-01 23:59:59', freq='15min', tz='Europe/Amsterdam')
        series = pd.Series(
            index=date_range,
            data=[0, 1] * int((len(date_range) / 2))
        )
        return heatmapfigure(series, rasterized=True)

    @compare_to_baseline
    def test_sinus(self):
        date_range = pd.date_range('2020-01-01', '2021-12-31 23:59:59', freq='15min', tz='UTC')
        data = date_range.to_series().apply(sinus_pattern)
        series = pd.Series(index=date_range, data=data)
        series.index = series.index.tz_convert('Europe/Zurich')
        return heatmapfigure(series, rasterized=True, annotate_suntimes=(47.492, 8.555))

    @compare_to_baseline
    def test_resolution_min(self):
        date_range = pd.date_range('2020-01-01', '2020-12-31 23:59:59', freq='60min', tz='UTC')
        # data = ([0, 1] * 12 + [1, 0] * 12) * 183
        data = date_range.to_series().apply(sinus_pattern, binary=False)
        series = pd.Series(index=date_range, data=data)
        return heatmapfigure(series, rasterized=True)

    @compare_to_baseline
    def test_resolution_max(self):
        # date_range = pd.date_range('2020-01-01', '2020-01-02 23:59:59', freq='1min', tz='UTC')
        # data = ([0, 1] * 60 * 12 + [1, 0] * 60 * 12)
        date_range = pd.date_range('2020-01-01', '2020-12-31 23:59:59', freq='1min', tz='UTC')
        data = date_range.to_series().apply(sinus_pattern, binary=False)
        series = pd.Series(index=date_range, data=data)
        return heatmapfigure(series, rasterized=True)

    @compare_to_baseline
    def test_less_than_one_day(self):
        date_range = pd.date_range('2020-01-01 12:15:00', '2020-01-01 23:19:59', freq='15min', tz='Europe/Amsterdam')
        series = pd.Series(
            index=date_range,
            data=[0, 1] * int((len(date_range) / 2)) + [0]
        )
        with self.assertWarnsRegex(RuntimeWarning, "Mean of empty slice") as cm:
            fig = heatmapfigure(series, rasterized=True)
        self.assertEqual(len(cm.warnings), 2)
        for warning in cm.warnings:
            self.assertEqual(str(warning.message), "Mean of empty slice")
        return fig

    @compare_to_baseline
    def test_empty_rc_params(self):
        with plt.rc_context({
            "savefig.bbox": "tight",
            "savefig.pad_inches": 1,
            "font.family": "serif",
        }):
            old_rc_params = HeatmapFigure.rc_params
            HeatmapFigure.rc_params = {}
            date_range = pd.date_range('2020-01-01', '2021-12-31 23:59:59', freq='15min', tz='UTC')
            data = date_range.to_series().apply(sinus_pattern)
            series = pd.Series(index=date_range, data=data)
            fig = heatmapfigure(series, rasterized=True)
            # This call here is required so the rc_params are set for savefig
            fig.savefig(f'{PATH_TEST_OUTPUT}/test_empty_rc_params.png')
            HeatmapFigure.rc_params = old_rc_params
        return None

    def test_invalid_series(self):
        date_range = pd.date_range('2020-01-01', '2020-12-31 23:59:59', freq='15min', tz='UTC')
        series = pd.Series(index=date_range, data=[0, 1] * int(len(date_range) / 2))
        series.index = series.index.tz_localize(None)
        with self.assertRaises(ValueError) as cm:
            heatmapfigure(series, rasterized=True)
        self.assertEqual(str(cm.exception), "Series index must be timezone-aware")

    @compare_to_baseline
    def test_daily_func(self):
        date_range = pd.date_range('2020-01-01', '2020-12-31 23:59:59', freq='15min', tz='UTC')
        series = pd.Series(index=date_range, data=[0, 1] * int(len(date_range) / 2))
        fig = heatmapfigure(series, rasterized=True, daily_func="mean", daily_label="Average (kW)")
        return fig

    def test_custom_attributes(self):
        # These are advertised public attributes of the class
        date_range = pd.date_range('2020-01-01', '2020-12-31 23:59:59', freq='15min', tz='UTC')
        series = pd.Series(index=date_range, data=[0, 1] * int(len(date_range) / 2))
        fig = heatmapfigure(series, rasterized=True)
        self.assertIsInstance(fig, HeatmapFigure)
        self.assertIsInstance(fig.rc_params, dict)
        self.assertIsInstance(fig.ax_cbar, plt.Axes)
        self.assertIsInstance(fig.ax_heatmap, plt.Axes)
        self.assertIsInstance(fig.ax_daily, plt.Axes)
        self.assertIsInstance(fig.ax_daily_peak, plt.Axes)
        self.assertIsInstance(fig.ax_hourly, plt.Axes)
        return None

    @compare_to_baseline
    def test_customization(self):
        # remove ticks, set x ticks manually
        date_range = pd.date_range('2020-01-01', '2021-12-31 23:59:59', freq='15min', tz='UTC')
        data = date_range.to_series().apply(sinus_pattern)
        series = pd.Series(index=date_range, data=data)
        series.index = series.index.tz_convert('Europe/Zurich')
        fig = heatmapfigure(series, rasterized=True, annotate_suntimes=(47.492, 8.555))
        fig.ax_heatmap.xaxis.set_major_locator(mdates.YearLocator())
        fig.ax_heatmap.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.ax_heatmap.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=1, bymonth=range(2, 13), tz=series.index.tz))
        fig.ax_heatmap.xaxis.set_minor_formatter(mdates.DateFormatter("%m", tz=series.index.tz))
        fig.ax_heatmap.xaxis.set_tick_params(which='minor', labelsize=6)
        fig.ax_heatmap.xaxis.set_tick_params(which='major', labelsize=8, length=4)
        fig.ax_heatmap.xaxis.get_majorticklabels()[-1].set_visible(False)
        return fig


class TestSuntimes(unittest.TestCase):
    def test_day_time_timezone_from_datetime(self):
        date = datetime.datetime(2010, 6, 21, 0, 6, 0, 0, tzinfo=datetime.timezone(
            offset=datetime.timedelta(hours=-7))
        )
        date, time, timezone = Sun.day_time_timezone_from_dateteime(date)
        self.assertEqual(date, 40350)
        self.assertAlmostEqual(time, 0.1 / 24, places=10)
        self.assertEqual(timezone, -7)

    def test_sun(self):
        s = Sun(40, -105)
        date = datetime.datetime(2010, 6, 21, 0, 6, 0, 0, tzinfo=datetime.timezone(
            offset=datetime.timedelta(hours=-7))
        )
        sunrise, sunset = s.get_suntimes(date)
        self.assertEqual(sunrise, datetime.time(4, 31, 16))
        self.assertEqual(sunset, datetime.time(19, 32, 9))


if __name__ == "__main__":
    unittest.main()
