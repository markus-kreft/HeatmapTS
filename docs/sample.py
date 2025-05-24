import pandas as pd
from heatmapts import heatmapfigure, HeatmapFigure
import matplotlib.dates as mdates
from cycler import cycler


THEMES = [
    {"name": "light",
     "color_foreground": "#1f2328",
     "color_background": "#ffffff",
     "color_fill_between": "#999999"},
    {"name": "dark",
     "color_foreground": "#f0f6fc",
     "color_background": "#0D1117",
     "color_fill_between": "#eeeeee"},
]


def download_and_convert(year, filename_csv):
    """Download Excel file and convert relevant columns to csv.
    https://www.swissgrid.ch/en/home/customers/topics/energy-data-ch.html
    """
    url = f"https://www.swissgrid.ch/content/dam/dataimport/energy-statistic/EnergieUebersichtCH-{year}.xlsx"
    columns = {
        "Unnamed: 0": "timestamp",
        "Summe endverbrauchte Energie Regelblock Schweiz\n"
        "Total energy consumed by end users in the Swiss controlblock": "total_energy_consumed",
        "Summe produzierte Energie Regelblock Schweiz\n"
        "Total energy production Swiss controlblock": "total_energy_produced",
    }

    df = pd.read_excel(url, sheet_name="Zeitreihen0h15", usecols=columns.keys())
    df = df.loc[1:]
    df.rename(columns=columns, inplace=True)
    df.to_csv(filename_csv, index=False)


def load_csv(filename):
    df = pd.read_csv(filename)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d.%m.%Y %H:%M")
    df.set_index("timestamp", inplace=True)
    df.index = df.index.tz_localize("Europe/Zurich", ambiguous="infer", nonexistent="shift_forward")
    df.index = df.index - pd.Timedelta("15min")
    df = df.asfreq("15min", fill_value=pd.NA)
    series = df["total_energy_consumed"]
    return series


def plot_theme(theme, *args, **kwargs):
    color_foreground = theme["color_foreground"]
    color_background = theme["color_background"]
    color_fill_between = theme["color_fill_between"]
    HeatmapFigure.rc_params.update({
        "text.color": color_foreground,
        "axes.facecolor": color_background,
        "figure.facecolor": color_background,
        "axes.labelcolor": color_foreground,
        "axes.edgecolor": color_foreground,
        "xtick.color": color_foreground,
        "ytick.color": color_foreground,
        "xtick.labelcolor": color_foreground,
        "ytick.labelcolor": color_foreground,
        "axes.prop_cycle": cycler('color', [color_fill_between]),
    })
    HeatmapFigure.COLOR_SCATTER = color_foreground

    fig = heatmapfigure(*args, **kwargs)

    fig.ax_heatmap.xaxis.set_major_locator(mdates.YearLocator())
    fig.ax_heatmap.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.ax_heatmap.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=1))
    fig.ax_heatmap.xaxis.set_minor_formatter(mdates.DateFormatter("%m"))
    fig.ax_heatmap.xaxis.set_tick_params(which='minor', labelsize=6, pad=1)
    fig.ax_heatmap.xaxis.set_tick_params(which='major', labelsize=8, length=4, pad=6)

    return fig


if __name__ == "__main__":
    # download_and_convert(2023, "docs/2023.csv")
    # download_and_convert(2024, "docs/2024.csv")
    series1 = load_csv("docs/2023.csv")
    series2 = load_csv("docs/2024.csv")
    series = pd.concat([series1, series2], axis=0)
    for theme in THEMES:
        fig = plot_theme(
            theme,
            series,
            rasterized=True,
            annotate_suntimes=(47.37, 8.54),
            hourly_label="Profile    \n(kW)"
        )
        fig.ax_heatmap.text(
            -0.07, -0.18,
            "Data by Swissgrid: "
            "https://www.swissgrid.ch/en/home/customers/topics/energy-data-ch.html",
            transform=fig.ax_heatmap.transAxes,
            ha="left", va="bottom", fontsize=6, color="#aaaaaa")
        fig.savefig(f"docs/sample-{theme['name']}.png", dpi=500)
