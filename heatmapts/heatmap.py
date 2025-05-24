import warnings
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from cycler import cycler
from typing import Optional

from .suntimes import Sun


class HeatmapFigure(plt.Figure):
    COLOR_AXES = "#999999"
    COLOR_SCATTER = "#000000"
    COLOR_SUNTIMES = "#ffffff"
    _default_figwidth = 8
    rc_params = {
        "figure.figsize": (_default_figwidth, _default_figwidth / 1.618),
        "font.family": "sans-serif",
        "font.size": 9,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.edgecolor": COLOR_AXES,
        "xtick.color": COLOR_AXES,  # modifies both ticks and labels
        "ytick.color": COLOR_AXES,
        "xtick.labelcolor": "black",  # set labels back to black
        "ytick.labelcolor": "black",
        "lines.color": COLOR_AXES,
        "axes.prop_cycle": cycler('color', ["#999999"]),  # this sets the facecolor for fill_between
    }

    ax_cbar: Optional[matplotlib.axes.Axes] = None
    ax_heatmap: Optional[matplotlib.axes.Axes] = None
    ax_daily: Optional[matplotlib.axes.Axes] = None
    ax_daily_peak: Optional[matplotlib.axes.Axes] = None
    ax_hourly: Optional[matplotlib.axes.Axes] = None

    def savefig(self, *args, **kwargs):
        with plt.rc_context(self.rc_params):
            super().savefig(*args, **kwargs)


def heatmapfigure(
    series: pd.Series,
    cbar_label: str = "Power (kW)",
    daily_label: str = "Energy (kWh)",
    hourly_label: str = "Profile (kW)",
    daily_max: bool = True,
    daily_func: str = "integral",
    title: Optional[str] = None,
    annotate_suntimes: Optional[tuple[float, float]] = None,
    figsize: Optional[tuple[int, int]] = None,
    **kwargs,
) -> HeatmapFigure:
    """Makes a figure with heatmap, daily overview, and hourly profile.

    Args:
        series: Series with timezone-aware DatetimeIndex and frequency.
                Timestamps describe the start of the interval for which the
                value is valid.
        cbar_label: Label for the colorbar.
        daily_label: Label for the y-axis of the daily overview. This axis
                     shows the integral of the series values over the day,
                     i.e., miltiplied by the interval length.
        hourly_label: Label for the y-axis of the mean profile.
        daily_max: If True, the daily maximum is plotted as a scatter plot.
        daily_func: Function to use for the daily overview. Can be "integral"
                    for the integral over the day or "mean" for the mean.
        title: Title of the figure.
        annotate_suntimes: Tuple with latitude and longitude to annotate
                           sunrise and sunset time for that location.
        figsize: Tuple with width and height of the figure in inches.
        **kwargs: Additional keyword arguments passed to pcolormesh.

    Returns:
        Custom Figure subclass with the plot and axes as additional attributes.
    """

    # check is series
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    # Check if the index is a datetime index
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex")
    # Check if the index is timezone aware
    if series.index.tz is None:
        raise ValueError("Series index must be timezone-aware")
    # Check if the index has a fixed frequency
    if series.index.freq is None:
        raise ValueError("Series index must have a frequency")
    if daily_func not in ["integral", "mean"]:
        raise ValueError("`daily_func` must be either 'integral' or 'mean'")

    interval_minutes = series.index.freq.nanos / 60e9

    # Generate the pivoted heatmap and corresponding time and date range
    data, daterange, timerange = _heatmap_data(series)

    with plt.rc_context(HeatmapFigure.rc_params):
        # Set up the figure and axes
        fig: HeatmapFigure = plt.figure(FigureClass=HeatmapFigure, figsize=figsize)  # type: ignore
        if title is not None:
            fig.suptitle(title)
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=(7, 1),
            height_ratios=(2, 8),
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.01,
            hspace=0.01 * (fig.get_figwidth() / fig.get_figheight()),
        )
        ax = fig.add_subplot(gs[1, 0])
        ax_daily = fig.add_subplot(gs[0, 0])
        ax_hourly = fig.add_subplot(gs[1, 1])
        ax_cbar = ax_daily.inset_axes((1.0, 0, 0.035, 1))  # (1.055, 0, 0.035, 1)

        _plot_hists(
            daterange,
            timerange,
            data,
            ax_daily,
            ax_hourly,
            interval_minutes,
            daily_max=daily_max,
            daily_func=daily_func,
        )
        ax_daily.set_ylabel(daily_label)
        ax_hourly.set_xlabel(hourly_label)

        mesh = plot_pcolormesh(ax, daterange, timerange, data, **kwargs)

        # Add and style the colorbar
        cbar = fig.colorbar(mesh, cax=ax_cbar, label=cbar_label)
        cbar.outline.set_visible(False)  # type: ignore
        ax_cbar.tick_params(which="both", rotation=0, left=False, labelleft=False)
        ax_cbar.minorticks_on()
        # for t in ax_cbar.get_yticklabels():
        #     t.set_verticalalignment('center')
        ax_cbar.set_zorder(100)

        if annotate_suntimes:
            _annotate_suntimes(ax, daterange, coords=annotate_suntimes)

        fig.ax_cbar = ax_cbar
        fig.ax_heatmap = ax
        fig.ax_daily = ax_daily
        fig.ax_hourly = ax_hourly
        if daily_max:
            fig.ax_daily_peak = fig.axes[3]
            # equalize y-axis limits of cbar and peak hist
            fig.ax_daily_peak.set_ylim(ax_cbar.get_ylim())

        return fig


def _heatmap_data(
    series: pd.Series
) -> tuple[np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Get [day x hour] matrix and date-/timeranges from series
    """

    # Pad to start and end of day if the series is does not cover a full day
    if series.index.min().date() == series.index.max().date():
        new_index = pd.date_range(series.index.min().floor('D'),
                                  series.index.max().floor('D') + pd.Timedelta(days=1),
                                  freq=series.index.freq)[:-1]
        series = series.reindex(new_index, fill_value=np.nan)

    timezone = series.index.tz
    # Alternative: set date and time as multiindex (drop duplicates) and use unstack
    df = series.copy().to_frame(name='values')
    df["date"] = df.index.date
    df["time"] = df.index.time
    data = df.pivot_table(
        index="time",
        columns="date",
        values='values',
        # in local time there are duplicates, set them to nan
        aggfunc=lambda x: x if len(x) == 1 else np.nan,
        # aggfunc=lambda x: x.iloc[0],
        # aggfunc=lambda x: x.sum(skipna=False),
        dropna=False,
    )
    # Construct daterange with timezone
    daterange = data.columns.astype("datetime64[ns]").tz_localize(timezone)
    # Add one day to the end because pcolormesh requires edges.  When last date
    # is one with time change backwards, adding one in local time does not add
    # the day. Therefore, use naive timestamps and add timezone explicitly.
    daterange = pd.date_range(start=daterange.min().tz_localize(None),
                              end=daterange.max().tz_localize(None) + pd.Timedelta(days=1),
                              tz=timezone)
    # Construct timerange with frequency and timezone information
    timerange = pd.date_range(
        start="1970-01-01T00:00:00",
        end="1970-01-02T00:00:00",
        freq=f"{series.index.freq.nanos}ns",
        tz=timezone,
    )
    # Numpy needs float type with np.nan instead of pf.NA
    data = data.replace(pd.NA, np.nan).to_numpy()
    return data, daterange, timerange


def plot_pcolormesh(ax, daterange, timerange, data, **kwargs):
    """
    Plot the 2D demand profile
    Take a numpy matrix and indices and make a figure with heatmap and sum/avg
    """

    mesh = ax.pcolormesh(daterange, timerange, data, **kwargs)
    ax.set_xlim(daterange[0], daterange[-1])
    ax.invert_yaxis()

    locator = mdates.AutoDateLocator(tz=daterange.tz)
    formatter = mdates.ConciseDateFormatter(locator, tz=daterange.tz)
    ax.xaxis.set_major_locator(locator)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(mdates.MonthLocator(tz=daterange.tz))

    # Alternative format: '%#H' if os.name == 'nt' else '%-H'
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H", tz=timerange.tz))
    ax.yaxis.set_major_locator(mdates.AutoDateLocator(tz=timerange.tz))
    ax.yaxis.set_minor_locator(mdates.HourLocator(tz=timerange.tz))

    # Remove last xticklabel to avoid overlap with next axis
    ax.xaxis.get_majorticklabels()[-1].set_visible(False)
    # Also remove first and last yticklabel to avoid overlap and because is 00
    ax.yaxis.get_majorticklabels()[0].set_visible(False)
    ax.yaxis.get_majorticklabels()[-1].set_visible(False)

    ax.set_xlabel("Date")
    ax.set_ylabel("Hour")

    # Hide axes frame lines and set color
    ax.spines[["top", "right"]].set_visible(False)

    return mesh


def _plot_hists(
    daterange: pd.DatetimeIndex,
    timerange: pd.DatetimeIndex,
    data: np.ndarray,
    ax_daily: matplotlib.axes.Axes,
    ax_hourly: matplotlib.axes.Axes,
    interval_minutes: float,
    daily_max: bool,
    daily_func: str,
) -> None:
    # Daily max
    if daily_max:
        with warnings.catch_warnings(category=RuntimeWarning, action="ignore"):
            daily_max_draw = np.nanmax(data, axis=0)
            daily_min_draw = np.nanmin(data, axis=0)
            # take max if max larger than abs(min)
            daily_peak_draw = np.where(
                np.abs(daily_max_draw) >= np.abs(daily_min_draw),
                daily_max_draw,
                daily_min_draw,
            )
        twinx = ax_daily.twinx()
        # twinx.set_ylabel("Peak", labelpad=0)
        twinx.scatter(
            daterange[:-1] + dt.timedelta(hours=12),
            daily_peak_draw,
            color=HeatmapFigure.COLOR_SCATTER,
            s=1,
            linewidths=0,
        )
        twinx.set_ylim(0, None)

        twinx.spines[["top", "left", "bottom"]].set_visible(False)

        # Rotate in case they are long
        # # does not work in older mpl
        # # twinx.set_yticks(twinx.get_yticks())
        # # twinx.set_yticklabels(twinx.get_yticklabels(), rotation=90, va='center')
        # for t in twinx.get_yticklabels():
        #     t.set_verticalalignment('center')
        # remove ticks and labels because we use the ones from the coolorbar
        twinx.set_yticklabels([])
        twinx.yaxis.set_tick_params(size=0)

        # Add a line at 0 if the min is below zero
        if np.nanmin(daily_peak_draw) < 0:
            twinx.axhline(0, color=HeatmapFigure.COLOR_AXES, lw=0.5)

    # Daily sum
    if daily_func == "integral":
        daily_demand = np.nansum(data, axis=0) * interval_minutes / 60
    elif daily_func == "mean":
        daily_demand = np.nanmean(data, axis=0)
    ax_daily.fill_between(
        daterange[:-1] + dt.timedelta(hours=12),
        daily_demand,
        alpha=0.5,
    )
    ax_daily.set_xlim(daterange[0], daterange[-1])
    # Need to set the max here as well, else when removing the lower tick (below)
    # the limits get extended, since the ticks have not been rendered yet.
    ax_daily.set_ylim(min(0, daily_demand.min()), daily_demand.max())
    # Remove first label because it may overlap with heat map
    # ax_histx.yaxis.get_majorticklabels()[0].set_visible(False)

    ax_daily.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    # Mean profile
    ax_hourly.fill_betweenx(
        timerange[:-1] + dt.timedelta(minutes=interval_minutes) / 2,
        np.nanmean(data, axis=1),
        alpha=0.5,
    )
    ax_hourly.set_zorder(-1)  # Draw behind so labels from other axes can be on graph
    ax_hourly.set_yticklabels([])
    # If demand is larger than zero, always show from zero, else show from negative demand on
    ax_hourly.set_xlim(min(0, np.nanmean(data, axis=1).min()), None)
    ax_hourly.set_ylim(timerange[0], timerange[-1])
    ax_hourly.invert_yaxis()  # This has to be called after setting lims

    # Hide the ticks and labels
    ax_daily.get_xaxis().set_visible(False)
    ax_hourly.get_yaxis().set_visible(False)
    # Hide axes frame lines
    ax_daily.spines[["top", "right", "bottom"]].set_visible(False)
    ax_hourly.spines[["top", "right", "left"]].set_visible(False)
    ax_daily.minorticks_on()
    ax_hourly.minorticks_on()


def _annotate_suntimes(
        ax: matplotlib.axes.Axes,
        daterange: pd.DatetimeIndex,
        coords: tuple[float, float],
) -> None:
    """Plots lines for sunrise and sunset times."""
    # Shift daterange by half a day to match the heatmap
    daterange = daterange[:-1] + pd.Timedelta(hours=12)
    times = daterange.to_frame(index=False, name="date").merge(
        pd.DataFrame(columns=["date", "sunrise", "sunset"]), how="left", on="date"
    )
    lat, long = coords
    sun = Sun(latitude=lat, longitude=long)
    times[["sunrise", "sunset"]] = times.apply(
        lambda d: [
            # non-aware datetime in local timezone
            dt.datetime.combine(dt.datetime(year=1970, month=1, day=1).date(), x)
            for x in sun.get_suntimes(d["date"])
        ],
        axis="columns",
        result_type="expand",
    )
    # Suntimes are calcualted in the local timezone and need to be localized
    times["sunset"] = times["sunset"].dt.tz_localize(daterange.tz)
    times["sunrise"] = times["sunrise"].dt.tz_localize(daterange.tz)

    ax.plot(daterange, times["sunset"], color=HeatmapFigure.COLOR_SUNTIMES, lw=0.5)
    ax.plot(daterange, times["sunrise"], color=HeatmapFigure.COLOR_SUNTIMES, lw=0.5)
