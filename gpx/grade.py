#!/Users/parautenbach/anaconda/bin/python

import gpxpy
import gpxpy.gpx
from haversine import haversine

fi = open('Additional_Singletrack.gpx', 'r')
gi = gpxpy.parse(fi)
ti = gi.tracks[0]
si = ti.segments[0]
pi = si.points

i_prev = None
d_acc = 0
for i in pi:
    if i_prev:
        d = haversine((i_prev.latitude, i_prev.longitude), (i.latitude, i.longitude))*1000
        d_acc += d
        if not d:
            g = 0
        else:
            g = (i.elevation - i_prev.elevation)/d
        print('{},{},{},{}'.format(i.time, d_acc, i.elevation, g))
    i_prev = i
