#!/bin/python
# coding: utf-8

"""Calculate a number of metrics and draw plots of those, e.g. more interesting and useful grade plots."""

import os.path
import warnings
from argparse import ArgumentParser

import gpxpy
import gpxpy.gpx
import matplotlib.pylab as plt
from matplotlib import collections, colors
import numpy as np
import scipy.signal as signal
import seaborn as sns

from haversine import haversine

_GPX_HR_TAG = "hr"
_GPX_CADENCE_TAG = "cad"
# percentage spacing above and below where required
_PLOT_PADDING = 0.2
_PLOT_DPI = 300
# 4.2m/s ~= 15km/h
# 2.8 m/s ~= 10km/h
_CUT_OFF_SPEED = 2.8  # 4.2
# default cut-off for the filter
_FILTER_DEFAULT_CUT_OFF = 0.05
_FILTER_ALT_CUT_OFF = 0.02
_FILTER_ORDER = 5
_FONT_SIZE = "xx-small"
# basically, if the grade is +/-33%, use that for the maximum for the colour map (because that is crazy steep)
_GRADIENT_CLIPPING_FACTOR = 3/10


def setup_argparser():
    """Set up the command-line argument parser."""
    parser = ArgumentParser(description="Generate a smoothed elevation plot (PNG) using a colour gradient \
                                         and stepped grades to show general climbs and downhills.")
    parser.add_argument("-f", "--file", required=True, dest="filename", help="the GPX file to generate the plot for", metavar="FILE")
    parser.add_argument("-hr", "--heart-rate", dest="plot_heart_rate", action="store_true", help="generate a heart rate plot too")
    parser.add_argument("-s", "--speed", dest="plot_speed", action="store_true", help="generate a speed plot too")
    parser.add_argument("-c", "--cadence", dest="plot_cadence", action="store_true", help="generate a cadence plot too")
    parser.add_argument("-i", "--interactive", dest="interactive_plot", action="store_true",
                        help="open an interactive plot window besides saving the plot to file")
    parser.add_argument("-q", "--quiet", dest="quiet", action="store_true", help="run in quiet mode")
    return parser


def get_gpx(filename, quiet):
    """Get the contents of a GPX file."""
    if not quiet:
        print("Parsing file {}".format(os.path.abspath(filename)))
    with open(filename, 'r') as input_file:
        return gpxpy.parse(input_file)


def parse_gpx(gpx):
    """Parse GPX data to extract elevation, heart rate and other data."""
    track = gpx.tracks[0]
    segment = track.segments[0]
    points = segment.points

    point_prev = None
    data = {
                "times": [],
                "distances": [],
                "cumulative_distances": [],
                "elevations": [],
                "speed": [],
                "heart_rates": [],
                "cadences": []
                }
    for point in points:
        if point_prev:
            data["times"].append(point.time)
            # in meters
            delta_distance = haversine((point_prev.latitude, point_prev.longitude), (point.latitude, point.longitude))*1000
            data["distances"].append(delta_distance)
            current_speed = (delta_distance/1000)/float((point.time - point_prev.time).total_seconds()/3600)
            data["speed"].append(current_speed)
            # append the last valid elevation in case of a None or so
            data["elevations"].append(point.elevation or data["elevations"][-1])
        if point.extensions:
            for extension in point.extensions[0].getchildren():
                if extension.tag[-2:] == _GPX_HR_TAG:
                    data["heart_rates"].append(int(extension.text))
                if extension.tag[-3:] == _GPX_CADENCE_TAG:
                    data["cadences"].append(int(extension.text))
        point_prev = point

    # km
    data["cumulative_distances"] = np.cumsum(data["distances"])/1000

    return (track, data)


def get_filter(average_speed):
    """Create a filter to smooth the elevation data."""
    # TODO: smarter cut-off
    # pace = 1/average_speed
    # cut_off = pace/20
    # print(cut_off)
    # if average speed is roughly more than 15km/h then relax the filter cut-off to get smoother elevation data
    # walk/run: 0.05
    # cycle: 0.01 - cause it's faster
    cut_off = _FILTER_DEFAULT_CUT_OFF
    if average_speed >= _CUT_OFF_SPEED:
        cut_off = _FILTER_ALT_CUT_OFF
    return signal.butter(_FILTER_ORDER, cut_off)


def calculate_metrics(markers, data):
    """Calculate metrics using the smoothed elevation data and group it according to the first derivative."""
    metrics = {
                "grades": [],
                "heart_rate_averages": [],
                "speed_averages": [],
                "cadence_averages": [],
                "cadence_percentages": []
                }
    for (idx, _) in enumerate(markers[1:]):
        start = markers[idx]
        end = markers[idx + 1]
        delta_distance = sum(data["distances"][start:end])
        delta_elevation = data["elevations"][end] - data["elevations"][start]
        if delta_distance == 0:
            grade = 0
        else:
            grade = delta_elevation/delta_distance*100
        # off by one error
        metrics["grades"].extend(np.ones(end - start)*grade)
        metrics["heart_rate_averages"].extend(np.ones(end - start)*np.average(data["heart_rates"][start:end]))
        # TODO: consider an alternative metric such as a percentile
        # https://stackoverflow.com/questions/29688168/mean-nanmean-and-warning-mean-of-empty-slice
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            metrics["cadence_averages"].extend(np.ones(end - start)*np.average(data["cadences"][start:end]))
            cadence_percentage_pedaling = len([c for c in data["cadences"][start:end] if c > 0])/float(end - start)
            metrics["cadence_percentages"].extend(np.ones(end - start)*cadence_percentage_pedaling)
        # in km/h
        average_speed = ((data["cumulative_distances"][end] - data["cumulative_distances"][start])/
                         float((data["times"][end] - data["times"][start]).total_seconds()/3600))
        metrics["speed_averages"].extend(np.ones(end - start)*average_speed)
    return metrics


def get_figure(args, heart_rates, cadences):
    """Create a subplot figure given the command-line arguments."""
    rows = None
    axes = None
    ax_elevation = None
    ax_speed = None
    ax_cadence = None
    ax_hr = None
    # speed, hr, cad
    # 1, 1, 1
    if args.plot_speed and (args.plot_heart_rate and heart_rates) and (args.plot_cadence and cadences):
        rows = 4
        (fig, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_speed, ax_hr, ax_cadence) = axes
    # speed, hr, not cad
    # 1, 1, 0
    elif args.plot_speed and (args.plot_heart_rate and heart_rates) and not (args.plot_cadence and cadences):
        rows = 3
        (fig, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_speed, ax_hr) = axes
    # speed, not hr, cad
    # 1, 0, 1
    elif args.plot_speed and not (args.plot_heart_rate and heart_rates) and (args.plot_cadence and cadences):
        rows = 3
        (fig, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_speed, ax_cadence) = axes
    # speed, not hr, not cad
    # 1, 0, 0
    elif args.plot_speed and not (args.plot_heart_rate and heart_rates) and not (args.plot_cadence and cadences):
        rows = 2
        (fig, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_speed) = axes
    # not speed, hr, cad
    # 0, 1, 1
    elif not args.plot_speed and (args.plot_heart_rate and heart_rates) and (args.plot_cadence and cadences):
        rows = 3
        (fig, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_hr, ax_cadence) = axes
    # not speed, hr, not cad
    # 0, 1, 0
    elif not args.plot_speed and (args.plot_heart_rate and heart_rates) and not (args.plot_cadence and cadences):
        rows = 2
        (fig, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_hr) = axes
    # not speed, not hr, cad
    # 0, 0, 1
    elif not args.plot_speed and not (args.plot_heart_rate and heart_rates) and args.plot_cadence:
        rows = 2
        (fig, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_cadence) = axes
    # not speed, not hr, not cad
    # 0, 0, 0
    else:
        rows = 1
        (fig, axes) = plt.subplots(rows, 1, sharex=True)
        ax_elevation = axes
    return (fig, (ax_elevation, ax_speed, ax_hr, ax_cadence))


def main():
    """Main programme."""
    # TODO: main()
    # TODO: specify time x-axis
    # https://stackoverflow.com/questions/1574088/plotting-time-in-python-with-matplotlib
    parser = setup_argparser()
    args = parser.parse_args()

    gpx = get_gpx(args.filename, args.quiet)
    (track, data) = parse_gpx(gpx)

    duration = gpx.get_moving_data().moving_time
    distance = gpx.get_moving_data().moving_distance
    average_speed = distance/duration

    butterworth_filter = get_filter(average_speed)
    elevations_filtered = signal.filtfilt(butterworth_filter[0], butterworth_filter[1], data["elevations"])
    gradient = np.diff(elevations_filtered)
    zero_crossings = np.where(np.diff(np.sign(gradient)))[0]

    markers = np.insert(zero_crossings, 0, 0)
    markers = np.append(markers, len(data["elevations"]) - 1)

    if not args.quiet:
        print("Calculating metrics")

    metrics = calculate_metrics(markers, data)

    if not args.quiet:
        print("Plotting")
    if not args.quiet and (args.plot_heart_rate and not data["heart_rates"]):
        print("WARNING: Heart rate plot requested but no heart rate data could be found")
    if not args.quiet and (args.plot_cadence and not data["cadences"]):
        print("WARNING: Cadence plot requested but no cadence data could be found")

    # https://www.codecademy.com/articles/seaborn-design-i
    sns.set_style(style="ticks", rc={"grid.linestyle": "--"})
    sns.set_context(rc={"grid.linewidth": 0.3})
    (_blue, orange, green, red, purple, _brown, _magenta, _grey, yellow, cyan) = sns.color_palette("deep")

    if args.interactive_plot:
        plt.ion()
    else:
        plt.ioff()

    (fig, (ax_elevation, ax_speed, ax_hr, ax_cadence)) = get_figure(args, data["heart_rates"], data["cadences"])
    axes = tuple([a for a in (ax_elevation, ax_speed, ax_hr, ax_cadence) if a])

    # ax_elevation.plot(x, elevations, color=green, label="Raw Elevation", fillstyle="bottom")
    # TODO: gradients
    ax_elevation.fill_between(data["cumulative_distances"], data["elevations"], 0, color=green, alpha=0.5)

    # calculate the smoothed gradients and create a colour map for it
    gradient_abs = np.abs(gradient)
    # first stretch the data to be in the range [0,1]
    gradient_normalised = gradient_abs/gradient_abs.max()
    gradient_clipped = np.clip(gradient_normalised, 0, _GRADIENT_CLIPPING_FACTOR)/_GRADIENT_CLIPPING_FACTOR
    cmap = colors.ListedColormap(sns.color_palette([yellow, orange, red]).as_hex())
    ax_elevation.scatter(data["cumulative_distances"][:-1], elevations_filtered[:-1], c=cmap(gradient_clipped), s=0.1, edgecolor=None)

    # plot the smoothed elevations, coloured according to the smoothed gradient
    # https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/multicolored_line.html
    elevation_points = np.array([data["cumulative_distances"], elevations_filtered]).T.reshape(-1, 1, 2)
    elevation_segments = np.concatenate([elevation_points[:-1], elevation_points[1:]], axis=1)
    elevation_lines = collections.LineCollection(elevation_segments, cmap=cmap)
    elevation_lines.set_array(gradient_clipped)
    ax_elevation.add_collection(elevation_lines)
    # line = ax_e...
    # TODO: f.colorbar(line, ax=ax_elevation)

    # other plot stuffs
    ax_elevation.set_xlim(min(data["cumulative_distances"]), max(data["cumulative_distances"]))
    ax_elevation.set_xlabel("Distance (km)", fontsize=_FONT_SIZE)
    ax_elevation.set_ylim(0, max(data["elevations"])*(1 + _PLOT_PADDING))
    ax_elevation.set_ylabel("m", fontsize=_FONT_SIZE)
    ax_elevation.tick_params(labelsize=_FONT_SIZE)
    ax_elevation.grid()

    ax_grade = ax_elevation.twinx()
    ax_grade.set_xlim(min(data["cumulative_distances"]), max(data["cumulative_distances"]))
    grades_max = np.ceil(max([abs(g) for g in metrics["grades"]]))
    # make symmetric
    ax_grade.set_ylim(-grades_max*(1 + _PLOT_PADDING), grades_max*(1 + _PLOT_PADDING))
    ax_grade.plot(data["cumulative_distances"][:-1], np.array(metrics["grades"]), color=orange, alpha=0.7, label="Stepped Grade")
    ax_grade.set_ylabel("%", fontsize=_FONT_SIZE)
    ax_grade.tick_params(labelsize=_FONT_SIZE)
    ax_grade.grid()

    if ax_speed:
        ax_speed.set_xlim(min(data["cumulative_distances"]), max(data["cumulative_distances"]))
        ax_speed.set_ylim(min(metrics["speed_averages"])*(1 - _PLOT_PADDING), max(metrics["speed_averages"])*(1 + _PLOT_PADDING))
        ax_speed.plot(data["cumulative_distances"][:-1], np.array(metrics["speed_averages"]), color=cyan, label="Average Speed")
        ax_speed.plot(data["cumulative_distances"][:-1], data["speed"][:-1], color=cyan, alpha=0.3, label="Speed")
        ax_speed.set_xlabel("Distance (km)", fontsize=_FONT_SIZE)
        ax_speed.set_ylabel("km/h", fontsize=_FONT_SIZE)
        ax_speed.set_ylim(0, max(data["speed"])*(1 + _PLOT_PADDING))
        ax_speed.legend(loc="upper right", fontsize=_FONT_SIZE)
        ax_speed.tick_params(labelsize=_FONT_SIZE)
        ax_speed.grid()

    if ax_hr:
        ax_hr.set_xlim(min(data["cumulative_distances"]), max(data["cumulative_distances"]))
        ax_hr.set_ylim(min(metrics["heart_rate_averages"])*(1 - _PLOT_PADDING), max(metrics["heart_rate_averages"])*(1 + _PLOT_PADDING))
        ax_hr.plot(data["cumulative_distances"][:-1], np.array(metrics["heart_rate_averages"]), color=red, label="Average Heart Rate")
        ax_hr.plot(data["cumulative_distances"][:-1], data["heart_rates"][:-2], color=red, alpha=0.3, label="Heart Rate")
        ax_hr.set_xlabel("Distance (km)", fontsize=_FONT_SIZE)
        ax_hr.set_ylabel("BPM", fontsize=_FONT_SIZE)
        ax_hr.legend(loc="upper right", fontsize=_FONT_SIZE)
        ax_hr.tick_params(labelsize=_FONT_SIZE)
        ax_hr.grid()

    if ax_cadence:
        ax_cadence.set_xlim(min(data["cumulative_distances"]), max(data["cumulative_distances"]))
        # ax_cadence.set_ylim(min(cadence_averages)*(1 - _PLOT_PADDING), max(cadence_averages)*(1 + _PLOT_PADDING))
        # ax_cadence.plot(cumulative_distances[:-1], np.array(cadence_averages), color=purple, label="Average Cadence")
        ax_cadence.plot(data["cumulative_distances"][:-1], np.array(metrics["cadence_percentages"])*100, color=purple, label="Percentage Pedaling")
        ax_cadence.plot(data["cumulative_distances"][:-1], data["cadences"][:-2], color=purple, alpha=0.3, label="Cadence")
        ax_cadence.set_xlabel("Distance (km)", fontsize=_FONT_SIZE)
        ax_cadence.set_ylabel("RPM / %", fontsize=_FONT_SIZE)
        ax_cadence.set_ylim(0, max(data["cadences"])*(1 + _PLOT_PADDING))
        ax_cadence.legend(loc="upper right", fontsize=_FONT_SIZE)
        ax_cadence.tick_params(labelsize=_FONT_SIZE)
        ax_cadence.grid()

    # h1, l1 = ax_elevation.get_legend_handles_labels()
    handles_ax_grade, legend_ax_grade = ax_grade.get_legend_handles_labels()
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    ax_grade.legend(handles_ax_grade, legend_ax_grade, loc="upper right", fontsize=_FONT_SIZE)

    fig.align_ylabels(axes)

    if track.name:
        # TODO: time
        fig.suptitle("{} on {}".format(track.name.strip(), track.get_time_bounds().start_time.date()))

    (input_basename, _) = os.path.splitext(os.path.basename(args.filename))
    output_filename = ".".join([input_basename, "png"])
    if not args.quiet:
        print("Saving plot to {}".format(os.path.abspath(output_filename)))
    plt.savefig(output_filename, dpi=_PLOT_DPI)

    if args.interactive_plot:
        plt.show()
        input("Press any key to quit ...")
        plt.close()

    # TODO: y-axis for cadence plot
    # TODO: Test with Jupyter
    # TODO: --all option, --cut-off option, --html
    # TODO: gradient plot
    # TODO: number of ups/downs and other summary stats, steepest ascent/descent

if __name__ == "__main__":
    main()
