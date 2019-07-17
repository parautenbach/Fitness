import gpxpy
import gpxpy.gpx
import scipy.signal as signal
import numpy as np
import matplotlib.pylab as plt

from haversine import haversine
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename", help="Input GPX FILE", metavar="FILE")
args = parser.parse_args()

with open(args.filename, 'r') as input_file:
    gpx = gpxpy.parse(input_file)

track = gpx.tracks[0]
segment = track.segments[0]
points = segment.points

point_prev = None
elevations = []
distances = []
for point in points:
    if point_prev:
        # in meters
        delta_distance = haversine((point_prev.latitude, point_prev.longitude), (point.latitude, point.longitude))*1000
        distances.append(delta_distance)
        elevations.append(point.elevation)
    point_prev = point

butterworth_filter = signal.butter(5, 0.005)
elevations_filtered = signal.filtfilt(butterworth_filter[0], butterworth_filter[1], elevations)
gradient = np.diff(elevations_filtered)
zero_crossings = np.where(np.diff(np.sign(gradient)))[0]

markers = np.insert(zero_crossings, 0, 0)
markers = np.append(markers, len(elevations) - 1)

grades = []
for (idx, marker) in enumerate(markers[1:]):
    start = markers[idx]
    end = markers[idx + 1]
    d = sum(distances[start:end])
    e = elevations[end] - elevations[start]
    grade = e/d*100
    # off by one error
    grades.extend(np.ones(end - start)*grade)
    #grades.append(grade)
    #print(grade)

x = np.cumsum(distances)/1000

plt.clf()
plt.grid()

ax1 = plt.gca()
ax1.plot(x, elevations, 'g', label='Raw elevation')
ax1.plot(x, elevations_filtered, 'b', label='Filtered elevation')
ax1.set_xlim(min(x), max(x))
ax1.set_ylim(0, 1.2*max(elevations))
ax1.set_xlabel('Distance (km)')
ax1.set_ylabel('Elevation / change (m)')

ax2 = ax1.twinx()
ax2.set_xlim(min(x), max(x))
ax2.plot(x[:-1], gradient, 'm', label='Gradient')
ax2.plot(x[zero_crossings], np.zeros(len(zero_crossings)), 'ko', label='Elevation changes')
ax2.plot(x[:-1], np.array(grades), 'r', label='Stepped Grade')
ax1.set_ylabel('Grade (%)')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize='x-small')

plt.savefig('output.png')
