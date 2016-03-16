
import math, time, itertools
import numpy as np
import os

import zmqmsgbus
import convex_area_localization
from convex_area_localization import *

PLOT = True # activate plot - this reduce drastically the computation rate
PRINT = True # print position

if PLOT:
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg


TIM561_START_ANGLE = -0.7853981634  # in rad,  = -45°
TIM561_STOP_ANGLE  =  3.926990817   # in rad,  = 225°

MAX_DIST_POINT = 0.005  # in cm
NB_LINE = 5             # number of line to search in points cloud

def PlotPolar():
    # Add polar grid lines
    plot.addLine(x=0, pen=0.2)
    plot.addLine(y=0, pen=0.2)

    radius = np.arange(0.2, 3, 0.2);
    for r in radius:
        circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, r*2, r*2)
        circle.setPen(pg.mkPen(color=(30, 30, 30)))
        plot.addItem(circle)

    radius = np.arange(1, 3.1, 1);
    for r in radius:
        circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, r*2, r*2)
        circle.setPen(pg.mkPen(color=(60, 60, 60)))
        plot.addItem(circle)
        plot.setXRange(-3, 3)
        plot.setYRange(-3, 3)

def pol2cart(radius, theta):
    ''' Convert polar coordinate onto cartesian coordinate

    Parameters are (N) numpy array of float
    Return cartesian (N,2) numpy array of float '''

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    cartesian = np.array([x, y]).T;
    return cartesian;

def update_scan_data(topic, message):
    global datagram
    datagram = message

def update_scan_pos(topic, message):
    global datagram_pos
    datagram_pos = message

# PyQtGraph stuff
if PLOT:
    app = QtGui.QApplication([])
    pg.setConfigOptions(antialias=True)
    plot = pg.plot(title='Lidar Polar Plot')
    plot.resize(600,600)
    plot.setAspectLocked()

# zmqmsgbus subscribe
bus = zmqmsgbus.Bus(sub_addr='ipc://ipc/source',
                    pub_addr='ipc://ipc/sink')
node = zmqmsgbus.Node(bus)

datagram = node.recv('/lidar/scan')
node.register_message_handler('/lidar/scan', update_scan_data)
datagram_pos = node.recv('/odometry/position')
node.register_message_handler('/odometry/position', update_scan_pos)

print('receiving')
while 1:
    position = np.asarray(datagram_pos)
    radius = np.asarray(datagram['Data'])
    theta = np.linspace(TIM561_START_ANGLE, TIM561_STOP_ANGLE, len(radius))

    cloud_pts = pol2cart(radius, theta)
    red_cloud_pts = density_reduction(cloud_pts, MAX_DIST_POINT, 1)
    red_cloud_pts = keep_border_points(red_cloud_pts)


    # find lines
    lines_model = find_lines(red_cloud_pts, NB_LINE)

    # convert into segment
    segments = segment_line(lines_model)

    # keep external segment
    ext_segments = keep_external_segment(segments)
    
    # extract valide intersection
    corners = extract_corner(ext_segments)

    # localize the robot using the intersection found
    pos, orientation = localize(corners, position)
    if pos is not None and orientation is not None:
        if PRINT:
            os.system('clear')
            print(str(np.mean(pos, axis=0))+" "+str(np.mean(orientation)))

        node.publish('/lidar/position', np.mean(pos, axis=0).tolist())
        node.publish('/lidar/orientation', np.mean(orientation))


    # draw raw measure
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    
    if PLOT:
        plot.clear()
        PlotPolar()

        # draw lidar points
        plot.plot(cloud_pts[:,0],cloud_pts[:,1],pen=None,symbol='o',
                  symbolPen=None,symbolSize=7,symbolBrush=(255,255,255,50))


        plot.plot(red_cloud_pts[:,0],red_cloud_pts[:,1],pen=None,symbol='o',
                  symbolPen=None,symbolSize=7,symbolBrush=(125,125,0,50))


        convexHulls = np.asarray(convex_hull(cloud_pts.tolist()))
        plot.plot(convexHulls[:,0],convexHulls[:,1],pen=None,symbol='o',
                  symbolPen=None,symbolSize=7,symbolBrush=(255,0,0,255))

        linePen = pg.mkPen(color=(200,200,200,200),width=2,
            style=QtCore.Qt.DotLine)
        colors=[(255,0,0,100),(0,255,0,100),(0,0,255,100),(255,255,0,100),
                (255,0,255,100),(0,127,255,100),(127,0,0,100),(0,127,0,100),
                (0,0,127,100),(127,255,0,100),(127,0,255,100),(0,127,255,100)]


        colors2=[(255,0,0,255),(0,255,0,255),(0,0,255,255),(255,255,0,255),
                (255,0,255,255),(0,127,255,255),(127,0,0,255),(0,127,0,255),
                (0,0,127,255),(127,255,0,255),(127,0,255,255),(0,127,255,255)]

        # draw segment
        for idx, segment in enumerate(ext_segments):
            linePen = pg.mkPen(color=colors2[idx],width=4,
                style=QtCore.Qt.SolidLine)
            plot.plot((segment.pt_A[0],segment.pt_B[0]),
                        (segment.pt_A[1],segment.pt_B[1]), pen=linePen)

        # draw table estimation corner
        table_corner = np.array([[0,0],[0,3],[2,3],[2,0]], dtype=float)
        table_corner = table_corner - [position[0], position[1]]
        table_corner = rotatePolygon(table_corner, -position[2]+math.pi/2)

        symbolePen = pg.mkPen(color=(255,255,255,255), width= 2)
        plot.plot(table_corner[:,0], table_corner[:,1], 
                pen=None,  symbol='x', symbolPen=symbolePen, symbolSize=12)

        # draw corners found
        symbolePen = pg.mkPen(color=(0,0,255,255), width= 2)
        if corners is not None:
            corner_x = [corner.x for corner in corners]
            corner_y = [corner.y for corner in corners]
            plot.plot(corner_x, corner_y, pen=None, symbol='x',
                symbolPen=symbolePen)

        

        app.processEvents()


