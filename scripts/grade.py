#!/bin/python
# coding: utf-8

import gpxpy
import gpxpy.gpx
import matplotlib.pylab as plt
import numpy as np
import os.path
import scipy.signal as signal
import seaborn as sns
import warnings

from argparse import ArgumentParser
from haversine import haversine
from matplotlib import colors
from matplotlib import cm
from matplotlib import collections

_GPX_HR_TAG = "hr"
_GPX_CADENCE_TAG = "cad"
# percentage spacing above and below where required
_PLOT_PADDING = 0.2
_PLOT_DPI = 300
# 4.2m/s ~= 15km/h
# 2.8 m/s ~= 10km/h
_CUT_OFF_SPEED = 2.8 # 4.2
# default cut-off for the filter
_FILTER_DEFAULT_CUT_OFF = 0.05
_FILTER_ALT_CUT_OFF = 0.03
_FILTER_ORDER = 5
_FONT_SIZE = "xx-small"
# basically, if the grade is +/-33%, use that for the maximum for the colour map (because that is crazy steep)
_GRADIENT_CLIPPING_FACTOR = 3/10

def setup_argparser():
    parser = ArgumentParser(description="Generate a smoothed elevation plot (PNG) using a colour gradient and stepped grades to show general climbs and downhills.")
    parser.add_argument("-f", "--file", required=True, dest="filename", help="the GPX file to generate the plot for", metavar="FILE")
    parser.add_argument("-hr", "--heart-rate", dest="plot_heart_rate", action="store_true", help="generate a heart rate plot too")
    parser.add_argument("-s", "--speed", dest="plot_speed", action="store_true", help="generate a speed plot too")
    parser.add_argument("-c", "--cadence", dest="plot_cadence", action="store_true", help="generate a cadence plot too")
    parser.add_argument("-i", "--interactive", dest="interactive_plot", action="store_true", help="open an interactive plot window besides saving the plot to file")
    parser.add_argument("-q", "--quiet", dest="quiet", action="store_true", help="run in quiet mode")
    return parser    

def get_gpx(filename, quiet):
    if not quiet:
        print("Parsing file {}".format(os.path.abspath(filename)))
    with open(filename, 'r') as input_file:
        return gpxpy.parse(input_file)

def parse_gpx(gpx):
    track = gpx.tracks[0]
    segment = track.segments[0]
    points = segment.points

    point_prev = None
    times = []
    elevations = []
    distances = []
    speed = []
    heart_rates = []
    cadences = []
    for point in points:
        if point_prev:
            times.append(point.time)
            # in meters
            delta_distance = haversine((point_prev.latitude, point_prev.longitude), (point.latitude, point.longitude))*1000
            distances.append(delta_distance)
            current_speed = (delta_distance/1000)/float((point.time - point_prev.time).total_seconds()/3600)
            speed.append(current_speed)
            # append the last valid elevation in case of a None or so
            elevations.append(point.elevation or elevations[-1])
        if point.extensions:
            for extension in point.extensions[0].getchildren():
                if extension.tag[-2:] == _GPX_HR_TAG:
                    heart_rates.append(int(extension.text))
                if extension.tag[-3:] == _GPX_CADENCE_TAG:
                    cadences.append(int(extension.text))
        point_prev = point
    return (track, times, distances, elevations, speed, heart_rates, cadences)

if __name__ == "__main__":
    # TODO: main()
    # TODO: specify time x-axis
    # https://stackoverflow.com/questions/1574088/plotting-time-in-python-with-matplotlib
    parser = setup_argparser()
    args = parser.parse_args()

    gpx = get_gpx(args.filename, args.quiet)
    (track, times, distances, elevations, speed, heart_rates, cadences) = parse_gpx(gpx)

    duration = gpx.get_moving_data().moving_time
    distance = gpx.get_moving_data().moving_distance
    average_speed = distance/duration

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
    butterworth_filter = signal.butter(_FILTER_ORDER, cut_off)
    elevations_filtered = signal.filtfilt(butterworth_filter[0], butterworth_filter[1], elevations)
    gradient = np.diff(elevations_filtered)
    zero_crossings = np.where(np.diff(np.sign(gradient)))[0]

    markers = np.insert(zero_crossings, 0, 0)
    markers = np.append(markers, len(elevations) - 1)

    # km
    x = np.cumsum(distances)/1000

    if not args.quiet:
        print("Calculating metrics")
    grades = []
    heart_rate_averages = []
    speed_averages = []
    cadence_averages = []
    cadence_percentages = []
    for (idx, marker) in enumerate(markers[1:]):
        start = markers[idx]
        end = markers[idx + 1]
        d = sum(distances[start:end])
        e = elevations[end] - elevations[start]
        if d == 0:
            grade = 0
        else:
            grade = e/d*100
        # off by one error
        grades.extend(np.ones(end - start)*grade)
        heart_rate_averages.extend(np.ones(end - start)*np.average(heart_rates[start:end]))
        # TODO: consider an alternative metric such as a percentile
        # https://stackoverflow.com/questions/29688168/mean-nanmean-and-warning-mean-of-empty-slice
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cadence_averages.extend(np.ones(end - start)*np.average(cadences[start:end]))
            cadence_percentage_pedaling = len([c for c in cadences[start:end] if c > 0])/float(end - start)
            cadence_percentages.extend(np.ones(end - start)*cadence_percentage_pedaling)
        # in km/h
        average_speed = (x[end] - x[start])/float((times[end] - times[start]).total_seconds()/3600)
        speed_averages.extend(np.ones(end - start)*average_speed)

    if not args.quiet:
        print("Plotting")
    # https://www.codecademy.com/articles/seaborn-design-i
    sns.set_style(style="ticks", rc={"grid.linestyle": "--"})
    sns.set_context(rc={"grid.linewidth": 0.3})
    (blue, orange, green, red, purple, brown, magenta, grey, yellow, cyan) = sns.color_palette("deep")

    if args.interactive_plot:
        plt.ion()
    else:
        plt.ioff()

    if args.plot_heart_rate and not heart_rates:
        print("WARNING: Heart rate plot requested but no heart rate data could be found")

    if args.plot_cadence and not cadences:
        print("WARNING: Cadence plot requested but no cadence data could be found")

    rows = None
    axes = None
    # speed, hr, cad
    # 1, 1, 1
    if args.plot_speed and (args.plot_heart_rate and heart_rates) and (args.plot_cadence and cadences):
        rows = 4
        (f, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_speed, ax_hr, ax_cadence) = axes
    # speed, hr, not cad
    # 1, 1, 0
    elif args.plot_speed and (args.plot_heart_rate and heart_rates) and not (args.plot_cadence and cadences):
        rows = 3
        (f, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_speed, ax_hr) = axes
    # speed, not hr, cad
    # 1, 0, 1
    elif args.plot_speed and not (args.plot_heart_rate and heart_rates) and (args.plot_cadence and cadences):
        rows = 3
        (f, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_speed, ax_cadence) = axes
    # speed, not hr, not cad
    # 1, 0, 0
    elif args.plot_speed and not (args.plot_heart_rate and heart_rates) and not (args.plot_cadence and cadences):
        rows = 2
        (f, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_speed) = axes
    # not speed, hr, cad
    # 0, 1, 1
    elif not args.plot_speed and (args.plot_heart_rate and heart_rates) and (args.plot_cadence and cadences):
        rows = 3
        (f, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_hr, ax_cadence) = axes
    # not speed, hr, not cad
    # 0, 1, 0
    elif not args.plot_speed and (args.plot_heart_rate and heart_rates) and not (args.plot_cadence and cadences):
        rows = 2
        (f, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_hr) = axes
    # not speed, not hr, cad
    # 0, 0, 1
    elif not args.plot_speed and not (args.plot_heart_rate and heart_rates) and args.plot_cadence:
        rows = 2
        (f, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_cadence) = axes
    # not speed, not hr, not cad
    # 0, 0, 0
    else:
        rows = 1
        (f, axes) = plt.subplots(rows, 1, sharex=True)
        ax_elevation = axes

    #ax_elevation.plot(x, elevations, color=green, label="Raw Elevation", fillstyle="bottom")
    # TODO: gradients
    ax_elevation.fill_between(x, elevations, 0, color=green, alpha=0.5)

    # calculate the smoothed gradients and create a colour map for it
    gg = np.abs(gradient)
    # first stretch the data to be in the range [0,1]
    gg = gg/gg.max()
    gg = np.clip(gg, 0, _GRADIENT_CLIPPING_FACTOR)/_GRADIENT_CLIPPING_FACTOR
    cmap = colors.ListedColormap(sns.color_palette([yellow, orange, red]).as_hex())
    sc = ax_elevation.scatter(x[:-1], elevations_filtered[:-1], c=cmap(gg), s=0.1, edgecolor=None)

    # plot the smoothed elevations, coloured according to the smoothed gradient
    # https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/multicolored_line.html
    elevation_points = np.array([x, elevations_filtered]).T.reshape(-1, 1, 2)
    elevation_segments = np.concatenate([elevation_points[:-1], elevation_points[1:]], axis=1)
    elevation_lines = collections.LineCollection(elevation_segments, cmap=cmap)
    elevation_lines.set_array(gg)
    line = ax_elevation.add_collection(elevation_lines)
    #f.colorbar(line, ax=ax_elevation)

    # other plot stuffs
    ax_elevation.set_xlim(min(x), max(x))
    ax_elevation.set_xlabel("Distance (km)", fontsize=_FONT_SIZE)
    ax_elevation.set_ylim(0, max(elevations)*(1 + _PLOT_PADDING))
    ax_elevation.set_ylabel("m", fontsize=_FONT_SIZE)
    ax_elevation.tick_params(labelsize=_FONT_SIZE)
    ax_elevation.grid()

    ax_grade = ax_elevation.twinx()
    ax_grade.set_xlim(min(x), max(x))
    grades_max = np.ceil(max([abs(g) for g in grades]))
    # symmetric
    ax_grade.set_ylim(-grades_max*(1 + _PLOT_PADDING), grades_max*(1 + _PLOT_PADDING))
    ax_grade.plot(x[:-1], np.array(grades), color=orange, alpha=0.7, label="Stepped Grade")
    ax_grade.set_ylabel("%", fontsize=_FONT_SIZE)
    ax_grade.tick_params(labelsize=_FONT_SIZE)
    ax_grade.grid()

    if args.plot_speed:
        ax_speed.set_xlim(min(x), max(x))
        ax_speed.set_ylim(min(speed_averages)*(1 - _PLOT_PADDING), max(speed_averages)*(1 + _PLOT_PADDING))
        ax_speed.plot(x[:-1], np.array(speed_averages), color=cyan, label="Average Speed")
        ax_speed.plot(x[:-1], speed[:-1], color=cyan, alpha=0.3, label="Speed")
        ax_speed.set_xlabel("Distance (km)", fontsize=_FONT_SIZE)
        ax_speed.set_ylabel("km/h", fontsize=_FONT_SIZE)
        ax_speed.set_ylim(0, max(speed)*(1 + _PLOT_PADDING))
        ax_speed.legend(loc="upper right", fontsize=_FONT_SIZE)
        ax_speed.tick_params(labelsize=_FONT_SIZE)
        ax_speed.grid()

    if args.plot_heart_rate and heart_rates:
        ax_hr.set_xlim(min(x), max(x))
        ax_hr.set_ylim(min(heart_rate_averages)*(1 - _PLOT_PADDING), max(heart_rate_averages)*(1 + _PLOT_PADDING))
        ax_hr.plot(x[:-1], np.array(heart_rate_averages), color=red, label="Average Heart Rate")
        ax_hr.plot(x[:-1], heart_rates[:-2], color=red, alpha=0.3, label="Heart Rate")
        ax_hr.set_xlabel("Distance (km)", fontsize=_FONT_SIZE)
        ax_hr.set_ylabel("BPM", fontsize=_FONT_SIZE)
        ax_hr.legend(loc="upper right", fontsize=_FONT_SIZE)
        ax_hr.tick_params(labelsize=_FONT_SIZE)
        ax_hr.grid()

    if args.plot_cadence and cadences:
        ax_cadence.set_xlim(min(x), max(x))
        #ax_cadence.set_ylim(min(cadence_averages)*(1 - _PLOT_PADDING), max(cadence_averages)*(1 + _PLOT_PADDING))
        #ax_cadence.plot(x[:-1], np.array(cadence_averages), color=purple, label="Average Cadence")
        ax_cadence.plot(x[:-1], np.array(cadence_percentages)*100, color=purple, label="Percentage Pedaling")
        ax_cadence.plot(x[:-1], cadences[:-2], color=purple, alpha=0.3, label="Cadence")
        ax_cadence.set_xlabel("Distance (km)", fontsize=_FONT_SIZE)
        ax_cadence.set_ylabel("RPM / %", fontsize=_FONT_SIZE)
        ax_cadence.set_ylim(0, max(cadences)*(1 + _PLOT_PADDING))
        ax_cadence.legend(loc="upper right", fontsize=_FONT_SIZE)
        ax_cadence.tick_params(labelsize=_FONT_SIZE)
        ax_cadence.grid()

    h1, l1 = ax_elevation.get_legend_handles_labels()
    h2, l2 = ax_grade.get_legend_handles_labels()
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    l = ax_grade.legend(h2, l2, loc="upper right", fontsize=_FONT_SIZE)

    f.align_ylabels(axes)

    if track.name:
        # TODO: time
        f.suptitle("{} on {}".format(track.name.strip(), track.get_time_bounds().start_time.date()))

    (input_basename, _) = os.path.splitext(os.path.basename(args.filename))
    output_filename = ".".join([input_basename, "png"])
    if not args.quiet:
        print("Saving plot to {}".format(os.path.abspath(output_filename)))
    plt.savefig(output_filename, dpi=_PLOT_DPI)

    if args.interactive_plot:
        plt.show()
        input("Press any key to quit ...")
        plt.close()

    # TODO: Test with Jupyter
    # TODO: --all option, --cut-off option, --html
    # TODO: gradient plot
    # TODO: number of ups/downs and other summary stats, steepest ascent/descent
