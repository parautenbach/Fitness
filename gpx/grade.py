import gpxpy
import gpxpy.gpx
import scipy.signal as signal
import numpy as np
import matplotlib.pylab as plt

from haversine import haversine
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename", help="Input GPX FILE", metavar="FILE")
# TODO: specify time x-axis
# https://stackoverflow.com/questions/1574088/plotting-time-in-python-with-matplotlib
# TODO: specify or extract activity type
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
        elevations.append(point.elevation)
    for extension in point.extensions[0].getchildren():
        if extension.tag[-2:] == 'hr':
            heart_rates.append(int(extension.text))
    point_prev = point

# walk/run: 0.05
# cycle: 0.01 – cause it's faster
butterworth_filter = signal.butter(5, 0.01)
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

#plt.clf()
#plt.grid()

(f, (ax1, ax3)) = plt.subplots(2, 1, sharex=True)

#ax1 = plt.gca()
ax1.plot(x, elevations, 'g', label='Raw elevation')
ax1.plot(x, elevations_filtered, 'b', label='Filtered elevation')
ax1.set_xlim(min(x), max(x))
#ax1.set_ylim(0, 1.2*max(elevations))
#ax1.set_xlabel('Distance (km)')
ax1.set_ylabel('Elevation / change (m)')
#ax1.legend(loc='upper right', fontsize='xx-small')
ax1.grid()

ax2 = ax1.twinx()
ax2.set_xlim(min(x), max(x))
#ax2.plot(x[:-1], gradient, 'k', label='Gradient')
#ax2.plot(x[zero_crossings], np.zeros(len(zero_crossings)), 'ko', label='Elevation changes')
ax2.plot(x[:-1], np.array(grades), 'm', label='Stepped Grade')
#ax2.set_xlabel('Distance (km)')
ax2.set_ylabel('Grade (%)')
#ax2.legend(loc='upper right', fontsize='xx-small')
ax2.grid()

# TODO: plot avg heart rate too ('r') / make optional arg
ax3.set_xlim(min(x), max(x))
ax3.set_ylim(0, 230)
ax3.plot(x[:-1], np.array(heart_rate_averages), 'r', label='Avg Heart Rate')
ax3.set_xlabel('Distance (km)')
ax3.set_ylabel('BPM')
ax3.legend(loc='upper right', fontsize='xx-small')
ax3.grid()

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize='x-small')

# TODO: specify filename
plt.savefig('output.png', dpi=300)
