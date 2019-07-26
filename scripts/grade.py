#!/bin/python
# coding: utf-8

import gpxpy
import gpxpy.gpx
import matplotlib.pylab as plt
import numpy as np
import os.path
import scipy.signal as signal
import seaborn as sns

from argparse import ArgumentParser
from haversine import haversine
from matplotlib import colors
from matplotlib import cm

_GPX_HR_TAG = "hr"
_GPX_CADENCE_TAG = "cad"
# percentage spacing above and below where required
_PLOT_PADDING = 0.2
_PLOT_DPI = 300
# default cut-off for the filter
_FILTER_DEFAULT_CUT_OFF = 0.03
_FILTER_ORDER = 5
_FONT_SIZE = "xx-small"

if __name__ == "__main__":
    # TODO: main()
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", required=True, dest="filename", help="the GPX file to generate the plot for", metavar="FILE")
    parser.add_argument("-hr", "--heart-rate", dest="plot_heart_rate", action="store_true", help="generate a heart rate plot too")
    parser.add_argument("-s", "--speed", dest="plot_speed", action="store_true", help="generate a speed plot too")
    parser.add_argument("-c", "--cadence", dest="plot_cadence", action="store_true", help="generate a cadence plot too")
    # TODO: specify time x-axis
    # https://stackoverflow.com/questions/1574088/plotting-time-in-python-with-matplotlib
    args = parser.parse_args()

    with open(args.filename, 'r') as input_file:
        gpx = gpxpy.parse(input_file)

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
    #if average_speed >= 4.2:
    #   cut_off = 0.01
    butterworth_filter = signal.butter(_FILTER_ORDER, cut_off)
    elevations_filtered = signal.filtfilt(butterworth_filter[0], butterworth_filter[1], elevations)
    gradient = np.diff(elevations_filtered)
    zero_crossings = np.where(np.diff(np.sign(gradient)))[0]

    markers = np.insert(zero_crossings, 0, 0)
    markers = np.append(markers, len(elevations) - 1)

    # km
    x = np.cumsum(distances)/1000

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
        cadence_averages.extend(np.ones(end - start)*np.average(cadences[start:end]))
        cadence_percentage_pedaling = len([c for c in cadences[start:end] if c > 0])/float(end - start)
        cadence_percentages.extend(np.ones(end - start)*cadence_percentage_pedaling)
        # in km/h
        average_speed = (x[end] - x[start])/float((times[end] - times[start]).total_seconds()/3600)
        speed_averages.extend(np.ones(end - start)*average_speed)

    # https://www.codecademy.com/articles/seaborn-design-i
    sns.set_style(style="ticks", rc={"grid.linestyle": "--"})
    sns.set_context(rc={"grid.linewidth": 0.3})
    (blue, orange, green, red, purple, brown, magenta, grey, yellow, cyan) = sns.color_palette("deep")

    rows = None
    axes = None
    # TODO: Other cadence options
    if args.plot_speed and (args.plot_heart_rate and heart_rates) and (args.plot_cadence and cadences):
        rows = 4
        (f, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_speed, ax_cadence, ax_hr) = axes
    elif (args.plot_heart_rate and heart_rates) and not args.plot_speed:
        rows = 2
        (f, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_hr) = axes
    elif args.plot_speed and not args.plot_heart_rate:
        rows = 2
        (f, axes) = plt.subplots(rows, 1, sharex=True)
        (ax_elevation, ax_speed) = axes
    else:
        rows = 1
        (f, axes) = plt.subplots(rows, 1, sharex=True)
        ax_elevation = axes

    #ax_elevation.plot(x, elevations, color=green, label="Raw Elevation", fillstyle="bottom")
    # TODO: gradients
    ax_elevation.fill_between(x, elevations, 0, color=green, alpha=0.5)
    #ax_elevation.plot(x, elevations_filtered, color=blue, label="Smoothed Elevation")
    #ax_elevation.scatter(x[:-1], elevations_filtered[:-1], c=np.abs(gradient), s=1, edgecolor=None, label="Smoothed Elevation")
    gg = np.abs(gradient)
    # first stretch the data to be in the range [0,1]
    gg = gg/gg.max()
    # then, basically, if the grade is +/-30%, use that for the maximum for the colour map
    s = 3/10
    gg = np.clip(gg, 0, s)/s
    cmap = colors.ListedColormap(sns.color_palette([yellow, orange, red]).as_hex())
    sc = ax_elevation.scatter(x[:-1], elevations_filtered[:-1], c=cmap(gg), s=0.1, edgecolor=None)
    ax_elevation.set_xlim(min(x), max(x))
    # TODO: refactor x-small (use rc or constant)
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

    if args.plot_heart_rate and not heart_rates:
        print("WARNING: Heart rate plot requested but no heart rate data could be found")

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

    if args.plot_cadence:
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

    # TODO: speed, cadence, gradient of gradient on elevation chart, just gradient
    # TODO: scatter plot: 
    # https://stackoverflow.com/questions/8453726/is-there-a-matplotlib-counterpart-of-matlab-stem3
    # fig = pyplot.figure(); ax = fig.add_subplot(111, projection='3d'); ax.scatter3D(heart_rates[0:-2:10], speed[0:-1:10], 100*ef[0::10])
    # ax.set_xlabel('hr'), ax.set_ylabel('s'), ax.set_zlabel('e')

    f.align_ylabels(axes)

    if track.name:
        # TODO: time
        f.suptitle("{} on {}".format(track.name.strip(), track.get_time_bounds().start_time.date()))

    (input_basename, _) = os.path.splitext(os.path.basename(args.filename))
    output_filename = ".".join([input_basename, "png"])
    plt.savefig(output_filename, dpi=_PLOT_DPI)
    plt.close()

    # TODO: number of ups/downs and other summary stats
