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
plt.plot(x, elevations, 'g', 
    x, elevations_filtered, 'b', 
    x[:-1], gradient*100, 'm', 
    x[zero_crossings], np.ones(len(zero_crossings)), 'ko',
    x[:-1], np.array(grades)*100, 'r')
plt.grid()
plt.xlim(min(x), max(x))
#plt.ylim(-2*max(elevations), 2*max(elevations))
plt.xlabel('Distance (km)')
plt.ylabel('Elevation / change (m)')
plt.legend(['Raw elevation','Filtered elevation','Gradient','Elevation changes','Stepped Grade'], fontsize='x-small')
plt.savefig('output.png')
