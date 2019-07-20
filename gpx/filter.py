#!/bin/python

import gpxpy
import gpxpy.gpx
from haversine import haversine

fi = open('activity_3745711592 copy.gpx', 'r')
gi = gpxpy.parse(fi)
ti = gi.tracks[0]
si = ti.segments[0]
pi = si.points

go = gpxpy.gpx.GPX()
to = gpxpy.gpx.GPXTrack()
go.tracks.append(to)
so = gpxpy.gpx.GPXTrackSegment()
to.segments.append(so)

p_new = []
i_prev = None
for i in pi:
    if i_prev:
        d = haversine((i_prev.latitude, i_prev.longitude), (i.latitude, i.longitude))*1000
        if d > 100:
            print(d)
            print(i)
            continue
    #so.points.append(pi)
    p_new.append(pi)
    i_prev = i

#x = go.to_xml()
