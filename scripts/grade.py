#!/bin/python
# coding: utf-8

"""Calculate a number of metrics and draw plots of those, e.g. more interesting and useful grade plots."""

import os.path
import sys
import warnings
from argparse import ArgumentParser
from datetime import datetime

import gpxpy
import gpxpy.gpx
import matplotlib.pylab as plt
from matplotlib import collections, colors, ticker
import numpy as np
import scipy.signal as signal
import seaborn as sns

from haversine import haversine
from tzlocal import get_localzone

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
    # positional
    parser.add_argument("filename", help="the GPX file to generate the plot for", metavar="filename")
    # optional
    parser.add_argument("-a", "--all", dest="plot_all", action="store_true", help="generate plots for all available and supported data and show the stats summary")
    parser.add_argument("-hr", "--heart-rate", dest="plot_heart_rate", action="store_true", help="generate a heart rate plot too")
    parser.add_argument("-s", "--speed", dest="plot_speed", action="store_true", help="generate a speed plot too")
    parser.add_argument("-c", "--cadence", dest="plot_cadence", action="store_true", help="generate a cadence plot too")
    parser.add_argument("-i", "--interactive", dest="interactive_plot", action="store_true",
                        help="open an interactive plot window besides saving the plot to file")
    parser.add_argument("-ss", "--summary", dest="show_summary", action="store_true", help="print some summary statistics for the workout")
    parser.add_argument("-q", "--quiet", dest="quiet", action="store_true", help="run in quiet mode")
    return parser


def get_gpx(filename, quiet):
    """Get the contents of a GPX file."""
    if not quiet:
        print("Parsing file {filename}".format(filename=os.path.abspath(filename)))
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
        # https://stackoverflow.com/questions/29688168/mean-nanmean-and-warning-mean-of-empty-slice
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            metrics["cadence_averages"].extend(np.ones(end - start)*np.average(data["cadences"][start:end]))
            cadence_percentage_pedaling = len([c for c in data["cadences"][start:end] if c > 0])/float(end - start)
            metrics["cadence_percentages"].extend(np.ones(end - start)*cadence_percentage_pedaling)
        # in km/h
        average_speed = ((data["cumulative_distances"][end] - data["cumulative_distances"][start]) /
                         float((data["times"][end] - data["times"][start]).total_seconds()/3600))
        metrics["speed_averages"].extend(np.ones(end - start)*average_speed)
    return metrics


def calculate_elevation_change(grade, distance):
    """Calculate the absolute elevation change."""
    return abs(grade/100)*(distance*1000)


def append_summary(group, summary, distance, grade, start, end):  # pylint: disable=too-many-arguments
    """Append summary data."""
    summary[group] = {
        "grade": grade,
        "distance": distance,
        "elevation_change": calculate_elevation_change(grade, distance),
        "section_start": start,
        "section_end": end
    }


def calculate_summary(data, metrics):
    """Calculate summary metrics."""
    summary = dict()
    if data["cadences"]:
        cadences_non_zero = [c for c in data["cadences"] if c > 0]
        overall_pedaling_fraction = len(cadences_non_zero) / float(len(data["cadences"]))
        cadence_average_non_zero = int(np.mean(cadences_non_zero))
        summary["overall_pedaling_fraction"] = overall_pedaling_fraction
        summary["cadence_average_non_zero"] = cadence_average_non_zero

    ascents = [g for g in np.unique(metrics["grades"]) if g > 0]
    summary["no_of_ascents"] = len(ascents)
    summary["average_ascent_grade"] = np.mean(ascents)

    descents = [g for g in np.unique(metrics["grades"]) if g < 0]
    summary["no_of_descents"] = len(descents)
    summary["average_descent_grade"] = np.mean(descents)

    # TODO: longest and steepest could match
    summary_list = {
        "steepest_ascent": np.max(metrics["grades"]),
        "longest_ascent": max(ascents, key=metrics["grades"].count),
        "steepest_descent": np.min(metrics["grades"]),
        "longest_descent": max(descents, key=metrics["grades"].count)
    }

    for (group, grade) in summary_list.items():
        (distance, (start, end)) = get_distance(grade, metrics["grades"], data["cumulative_distances"])
        append_summary(group, summary, distance, grade, start, end)

    return summary


def print_summary(summary):
    """Print summary data to the console."""
    print("\nSummary statistics:")
    if "overall_pedaling_fraction" in summary:
        print("  Overall pedaling percentage was {pp:.1%} at an average of {nzc} RPM".format(pp=summary["overall_pedaling_fraction"],
                                                                                             nzc=summary["cadence_average_non_zero"]))
        print()

    print("  {cnt} ascents at an average grade of {grade:.1f}%".format(cnt=summary["no_of_ascents"],
                                                                       grade=summary["average_ascent_grade"]))
    print("  Steepest ascent was {el:.0f}m over {dist:.3f}km with a grade of {grade:.1f}%".format(el=summary["steepest_ascent"]["elevation_change"],
                                                                                                  dist=summary["steepest_ascent"]["distance"],
                                                                                                  grade=summary["steepest_ascent"]["grade"]))
    print("  Longest ascent was {el:.0f}m over {dist:.3f}km with a grade of {grade:.1f}%".format(el=summary["longest_ascent"]["elevation_change"],
                                                                                                 dist=summary["longest_ascent"]["distance"],
                                                                                                 grade=summary["longest_ascent"]["grade"]))
    print()

    print("  {cnt} descents at an average grade of {grade:.1f}%".format(cnt=summary["no_of_descents"],
                                                                        grade=summary["average_descent_grade"]))
    print("  Steepest descent was {el:.0f}m over {dist:.3f}km with a grade of {grade:.1f}%".format(el=summary["steepest_descent"]["elevation_change"],
                                                                                                   dist=summary["steepest_descent"]["distance"],
                                                                                                   grade=summary["steepest_descent"]["grade"]))
    print("  Longest descent was {el:.0f}m over {dist:.3f}km at a grade of {grade:.1f}%".format(el=summary["longest_descent"]["elevation_change"],
                                                                                                dist=summary["longest_descent"]["distance"],
                                                                                                grade=summary["longest_descent"]["grade"]))
    # print()


def add_subplot(fig):
    """Add another subplot row and adjust the geometry accordingly."""
    n = len(fig.axes)
    for i in range(n):
        fig.axes[i].change_geometry(n + 1, 1, i + 1)

    return fig.add_subplot(n + 1, 1, n + 1)


def get_figure(args, heart_rates, cadences):
    """Create a subplot figure given the command-line arguments."""
    # https://gist.github.com/LeoHuckvale/89683dc242f871c8e69b
    ax_speed = None
    ax_hr = None
    ax_pedaling = None

    fig = plt.figure()
    ax_elevation = fig.add_subplot(1, 1, 1)

    if args.plot_speed:
        ax_speed = add_subplot(fig)

    if args.plot_heart_rate and heart_rates:
        ax_hr = add_subplot(fig)

    if args.plot_cadence and cadences:
        ax_pedaling = add_subplot(fig)

    return (fig, (ax_elevation, ax_speed, ax_hr, ax_pedaling))


def get_distance(grade, grades, cumulative_distances):
    """Get the distance covered by a specific grade."""
    start = grades.index(grade)
    end = len(grades) - 1 - grades[::-1].index(grade)
    distance = cumulative_distances[end] - cumulative_distances[start]
    return (distance, (start, end))


def mark_section_highlights(axis, distances, elevations, summary, colour):
    """Mark section highlights (steepest/longest ascent/descent) on the elevation plot."""
    for group in summary:
        if isinstance(summary[group], dict):
            start = summary[group]["section_start"]
            # skip the last one purely for aesthetic reasons
            end = summary[group]["section_end"] - 1
            axis.fill_between(distances[start:end], elevations[start:end], 0, color=colour, alpha=0.8)


def set_common_plot_options(axis, distances, legend=True):
    """Set plot options common to all axes."""
    axis.set_xlim(min(distances), max(distances))
    if legend:
        axis.legend(loc="upper right", fontsize=_FONT_SIZE)
    axis.tick_params(labelsize=_FONT_SIZE)
    axis.set_xticklabels([])
    axis.grid()


def plot_speed(axis, distances, speed_averages, speed, colour):
    """Plot speed data."""
    axis.plot(distances[:-1], np.array(speed_averages), color=colour, label="Average Speed")
    axis.plot(distances[:-1], speed[:-1], color=colour, alpha=0.3, label="Speed")
    axis.set_ylim(0, max(speed) * (1 + _PLOT_PADDING))
    axis.set_ylabel("km/h", fontsize=_FONT_SIZE)
    set_common_plot_options(axis, distances)


def plot_hr(axis, distances, heart_rate_averages, heart_rates, colour):
    """Plot heart rate data."""
    axis.plot(distances[:-1], np.array(heart_rate_averages), color=colour, label="Average Heart Rate")
    axis.plot(distances[:-1], heart_rates[:-2], color=colour, alpha=0.3, label="Heart Rate")
    axis.set_ylim(min(heart_rate_averages) * (1 - _PLOT_PADDING), max(heart_rate_averages) * (1 + _PLOT_PADDING))
    axis.set_ylabel("BPM", fontsize=_FONT_SIZE)
    set_common_plot_options(axis, distances)


def plot_pedaling(ax_pedaling, distances, cadence_percentages, cadences, colour):
    """Plot pedaling and cadence data."""
    cadence_percentages = np.array(cadence_percentages) * 100
    ax_pedaling.plot(distances[:-1], cadence_percentages, color=colour, label="Percentage Pedaling")
    # make symmetric
    ax_pedaling.set_ylim(0, max(cadences) * (1 + _PLOT_PADDING))
    ax_pedaling.set_ylabel("%", fontsize=_FONT_SIZE)
    set_common_plot_options(ax_pedaling, distances, legend=False)

    ax_cadence = ax_pedaling.twinx()
    ax_cadence.plot(distances[:-1], cadences[:-2], color=colour, alpha=0.3, label="Cadence")
    ax_cadence.set_xlabel("Distance (km)", fontsize=_FONT_SIZE)
    ax_cadence.set_ylabel("RPM", fontsize=_FONT_SIZE)
    ax_cadence.set_ylim(0, max(cadences) * (1 + _PLOT_PADDING))
    set_common_plot_options(ax_cadence, distances, legend=False)

    handles_ax_cadence, legend_ax_cadence = ax_cadence.get_legend_handles_labels()
    handles_ax_pedaling, legend_ax_pedaling = ax_pedaling.get_legend_handles_labels()
    ax_cadence.legend(handles_ax_pedaling + handles_ax_cadence, legend_ax_pedaling + legend_ax_cadence, loc="upper right", fontsize=_FONT_SIZE)


def plot_graded_elevation(axis, distances, elevations, gradient, colour_map):
    """Plot the elevation line using a colour gradient."""
    # plot the smoothed elevations, coloured according to the smoothed gradient
    # https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/multicolored_line.html
    elevation_points = np.array([distances, elevations]).T.reshape(-1, 1, 2)
    elevation_segments = np.concatenate([elevation_points[:-1], elevation_points[1:]], axis=1)
    elevation_lines = collections.LineCollection(elevation_segments, cmap=colour_map)
    elevation_lines.set_array(gradient)
    axis.add_collection(elevation_lines)
    # TODO: f.colorbar(line, ax=ax_elevation)


def get_elevation_ylim(elevations):
    """Determine a suitable limit for the y-axis that will ensure some even multiple of one of the ticks will be exactly halfway."""
    # do this to align the y-axes
    # not clear exactly how matplotlib determines ticks, but it seems to involve some even number, hence the 2
    # we do two calculations to cater for elevation ranges in the tens vs the hundreds
    if max(elevations)/200.0 > 0:
        return np.ceil(max(elevations) / 200) * 200

    return np.ceil(max(elevations) / 20) * 20


def plot_elevation(axis, distances, elevations, elevations_filtered, gradient, summary, colour, colour_range):  # pylint: disable=too-many-arguments
    """Plot elevation data."""
    axis.fill_between(distances, elevations, 0, color=colour, alpha=0.5)
    mark_section_highlights(axis, distances, elevations, summary, colour)
    # calculate the smoothed gradients and create a colour map for it
    gradient_abs = np.abs(gradient)
    # first stretch the data to be in the range [0,1]
    gradient_normalised = gradient_abs/gradient_abs.max()
    gradient_clipped = np.clip(gradient_normalised, 0, _GRADIENT_CLIPPING_FACTOR)/_GRADIENT_CLIPPING_FACTOR
    colour_map = colors.ListedColormap(sns.color_palette(colour_range).as_hex())
    axis.scatter(distances[:-1], elevations_filtered[:-1], c=colour_map(gradient_clipped), s=0.1, edgecolor=None)

    plot_graded_elevation(axis, distances, elevations_filtered, gradient, colour_map)

    axis.set_ylim(0, get_elevation_ylim(elevations))
    axis.set_ylabel("m", fontsize=_FONT_SIZE)

    set_common_plot_options(axis, distances, legend=False)


def plot_grades(axis, distances, grades, colour):
    """Plot grade data."""
    grades_max = np.ceil(max([abs(g) for g in grades]))
    # make symmetric
    axis.set_ylim(-grades_max*(1 + _PLOT_PADDING), grades_max*(1 + _PLOT_PADDING))
    axis.plot(distances[:-1], np.array(grades), color=colour, alpha=0.7, label="Stepped Grade")
    axis.set_ylabel("%", fontsize=_FONT_SIZE)
    set_common_plot_options(axis, distances, legend=False)
    axis.axhline(y=0, color=colour, alpha=0.5, linestyle="--", linewidth=0.5)


def save_figure(fig, args):
    """Save the figure to disk."""
    (input_basename, _) = os.path.splitext(os.path.basename(args.filename))
    output_filename = ".".join([input_basename, "png"])
    if not args.quiet:
        print("Saving plot to {filename}".format(filename=os.path.abspath(output_filename)))
    fig.savefig(output_filename, dpi=_PLOT_DPI)


def main():
    """Run the main programme."""
    parser = setup_argparser()
    args = parser.parse_args()

    if args.plot_all:
        args.plot_heart_rate = True
        args.plot_speed = True
        args.plot_cadence = True
        args.show_summary = True

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
    summary = calculate_summary(data, metrics)

    if not args.quiet:
        print("Plotting")
        if args.plot_heart_rate and not data["heart_rates"]:
            print("WARNING: Heart rate plot requested but no heart rate data could be found", file=sys.stderr)
        if args.plot_cadence and not data["cadences"]:
            print("WARNING: Cadence plot requested but no cadence data could be found", file=sys.stderr)

    # https://www.codecademy.com/articles/seaborn-design-i
    sns.set_style(style="ticks", rc={"grid.linestyle": "--"})
    sns.set_context(rc={"grid.linewidth": 0.3})
    (_blue, orange, green, red, purple, _brown, _magenta, _grey, yellow, cyan) = sns.color_palette("deep")

    if args.interactive_plot:
        plt.ion()
    else:
        plt.ioff()

    (fig, (ax_elevation, ax_speed, ax_hr, ax_pedaling)) = get_figure(args, data["heart_rates"], data["cadences"])
    axes = tuple([a for a in (ax_elevation, ax_speed, ax_hr, ax_pedaling) if a])

    plot_elevation(ax_elevation, data["cumulative_distances"], data["elevations"], elevations_filtered, gradient, summary, green, [yellow, orange, red])
    ax_grade = ax_elevation.twinx()
    plot_grades(ax_grade, data["cumulative_distances"], metrics["grades"], orange)
    handles_ax_grade, legend_ax_grade = ax_grade.get_legend_handles_labels()
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    ax_grade.legend(handles_ax_grade, legend_ax_grade, loc="upper right", fontsize=_FONT_SIZE)

    if ax_speed:
        plot_speed(ax_speed, data["cumulative_distances"], metrics["speed_averages"], data["speed"], cyan)
    if ax_hr:
        plot_hr(ax_hr, data["cumulative_distances"], metrics["heart_rate_averages"], data["heart_rates"], red)
    if ax_pedaling:
        plot_pedaling(ax_pedaling, data["cumulative_distances"], metrics["cadence_percentages"], data["cadences"], purple)

    bottom_axis = axes[-1]
    bottom_axis.xaxis.set_major_formatter(ticker.ScalarFormatter())
    bottom_axis.set_xlabel("Distance (km)", fontsize=_FONT_SIZE)
    fig.align_ylabels(axes)

    if track.name:
        # https://stackoverflow.com/questions/1111317/how-do-i-print-a-python-datetime-in-the-local-timezone
        # assume the gpx time is always UTC
        start_time = track.get_time_bounds().start_time.astimezone(get_localzone())
        fig.suptitle("{name} on {date} at {time}".format(name=track.name.strip(), date=start_time.date(), time=start_time.time().strftime("%H:%M")))
        # fig.suptitle("{name} on {date} at {time}".format(name=track.name.strip(), date=start_time.date().strftime("%x"), time=start_time.time().strftime("%X")))

    save_figure(fig, args)

    if args.interactive_plot:
        plt.show()
        input("Press any key to quit ...")
        plt.close()

    if args.show_summary:
        print_summary(summary)

    # TODO: Test with Jupyter
    # TODO: --cut-off option, --html
    # TODO: gradient plot
    # TODO: specify time x-axis / space axis according to time
    # https://stackoverflow.com/questions/1574088/plotting-time-in-python-with-matplotlib
    # https://realpython.com/python-code-quality/


if __name__ == "__main__":
    main()
