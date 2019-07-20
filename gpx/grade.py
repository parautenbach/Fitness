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

_GPX_HR_TAG = "hr"
# percentage spacing above and below where required
_PLOT_PADDING = 0.2
_PLOT_DPI = 300
# default cut-off for the filter
_FILTER_DEFAULT_CUT_OFF = 0.03
_FILTER_ORDER = 5

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", required=True, dest="filename", help="the GPX file to generate the plot for", metavar="FILE")
    parser.add_argument("-hr", "--heart-rate", dest="plot_heart_rate", action='store_true', help="generate a heart rate plot too")
    # TODO: specify time x-axis
    # https://stackoverflow.com/questions/1574088/plotting-time-in-python-with-matplotlib
    args = parser.parse_args()

    with open(args.filename, 'r') as input_file:
        gpx = gpxpy.parse(input_file)

    track = gpx.tracks[0]
    segment = track.segments[0]
    points = segment.points

    point_prev = None
    elevations = []
    distances = []
    heart_rates = []
    for point in points:
        if point_prev:
            # in meters
            delta_distance = haversine((point_prev.latitude, point_prev.longitude), (point.latitude, point.longitude))*1000
            distances.append(delta_distance)
            # append the last valid elevation in case of a None or so
            elevations.append(point.elevation or elevations[-1])
        if point.extensions:
            for extension in point.extensions[0].getchildren():
                if extension.tag[-2:] == _GPX_HR_TAG:
                    heart_rates.append(int(extension.text))
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

    grades = []
    heart_rate_averages = []
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

    x = np.cumsum(distances)/1000

    # https://www.codecademy.com/articles/seaborn-design-i
    sns.set_style(style="ticks", rc={'grid.linestyle': '--'})
    sns.set_context(rc={"grid.linewidth": 0.3})
    palette = sns.color_palette("deep")

    if args.plot_heart_rate and heart_rates:
        (f, (ax1, ax3)) = plt.subplots(2, 1, sharex=True)
    else:
        (f, ax1) = plt.subplots(1, 1, sharex=True)

    ax1.plot(x, elevations, color=palette[2], label='Raw Elevation', fillstyle="bottom")
    ax1.fill_between(x, elevations, 0, color=palette[2], alpha=0.5)
    ax1.plot(x, elevations_filtered, color=palette[0], label='Smoothed Elevation')
    ax1.set_xlim(min(x), max(x))
    ax1.set_ylim(0, max(elevations)*(1 + _PLOT_PADDING))
    ax1.set_ylabel('Elevation / change (m)')
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.set_xlim(min(x), max(x))
    ax2.plot(x[:-1], np.array(grades), color=palette[1], label='Stepped Grade')
    ax2.set_ylabel('Grade (%)')
    ax2.grid()

    if args.plot_heart_rate and not heart_rates:
        print("WARNING: Heart rate plot requested but no heart rate data could be found")

    if args.plot_heart_rate and heart_rates:
        ax3.set_xlim(min(x), max(x))
        # email me if you're alive if your heart rate exceeded 230 bpm
        ax3.set_ylim(min(heart_rate_averages)*(1 - _PLOT_PADDING), max(heart_rate_averages)*(1 + _PLOT_PADDING))
        ax3.plot(x[:-1], np.array(heart_rate_averages), color=palette[3], label='Average Heart Rate')
        ax3.set_xlabel('Distance (km)')
        ax3.set_ylabel('BPM')
        ax3.legend(loc='upper right', fontsize='xx-small')
        ax3.grid()

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3 , fontsize='xx-small')

    if track.name:
        # TODO: track.get_time_bounds().start_time + date
        f.suptitle(track.name)

    (input_basename, _) = os.path.splitext(os.path.basename(args.filename))
    output_filename = '.'.join([input_basename, "png"])
    plt.savefig(output_filename, dpi=_PLOT_DPI)
    plt.close()
