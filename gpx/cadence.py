#!/bin/python

import gpxpy
import gpxpy.gpx
from haversine import haversine

fi = open('2019-07-07_10-25_Waterfalls_and_proteas_Ride.gpx', 'r')
gi = gpxpy.parse(fi)
ti = gi.tracks[0]
si = ti.segments[0]
pi = si.points

cad = []
for i in pi:
    for e in i.extensions[0].getchildren():
        if e.tag[-3:] == 'cad':
            cad.append(int(e.text))

print(len([x for x in cad if x > 0])/float(len(cad)))
