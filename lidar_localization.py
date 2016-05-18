
import argparse
import json
from jsmin import jsmin
import math 
import time
import itertools
import numpy as np
import os

import zmqmsgbus
import convex_area_localization
from convex_area_localization import *

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('config', type=argparse.FileType('r'), help='File where the configuration is stored.')

    parser.add_argument('--plot', help='Plot LIDAR points. Activate this option reduce drastically the computation rate !', action="store_true")

    parser.add_argument('--print', dest='print_output', help='Print estimated postion and orientation in the shell.', action="store_true")

    return parser.parse_args()


def PlotPolar(plot):
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

    
def main():
    global datagram
    global datagram_pos

    args = parse_args()
    config = json.loads(jsmin(args.config.read()))

    # PyQtGraph stuff
    if args.plot:
        app = QtGui.QApplication([])
        pg.setConfigOptions(antialias=False)
        plot = pg.plot(title='Lidar Polar Plot')
        plot.resize(600,600)
        plot.setAspectLocked()

    # zmqmsgbus subscribe
    bus = zmqmsgbus.Bus(sub_addr='ipc://ipc/source',
                        pub_addr='ipc://ipc/sink')
    node = zmqmsgbus.Node(bus)

    datagram = node.recv('/lidar/scan')
    node.register_message_handler('/lidar/scan', update_scan_data)
    datagram_pos = node.recv('/position')
    node.register_message_handler('/position', update_scan_pos)

    datagram_pos = np.asarray([0.5,0.5,1.57])

    print('receiving')
    while 1:
        radius = np.asarray(datagram['Data'])
        theta = np.linspace(config['TIM561_START_ANGLE'], config['TIM561_STOP_ANGLE'], len(radius))
        cloud_pts = pol2cart(radius, theta)

        # reduce cloud point density
        red_cloud_pts = density_reduction(cloud_pts, config['MAX_DIST_POINT'], 1)

        # find lines
        lines_model = find_lines(red_cloud_pts, config['NB_LINE'])

        # convert lines onto segments
        segments = segment_line(lines_model)

        # only keep external segments
        ext_segments = keep_external_segment(segments)
        
        # extract valide intersections
        corners = extract_corner(ext_segments)

        # localize the robot using the intersections found
        positions, orientations = localize(corners, datagram_pos, config['TABLE_WIDTH'], config['TABLE_HEIGHT'])

        position = np.array([0,0])
        orientation = np.array([0])

        if positions is not None and orientations is not None:
            position = np.mean(positions, axis=0).squeeze()
            orientation = np.mean(orientations)

            if args.print_output:
                os.system('clear')
                print(str(position)+" "+str(orientation))

            node.publish('/lidar/position', np.mean(position, axis=0).tolist())
            node.publish('/lidar/orientation', np.mean(orientation))


        # draw raw measure
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        if args.plot:
            plot.clear()
            PlotPolar(plot)

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

            colors=[(255,0,0,255),(0,255,0,255),(0,0,255,255),(255,255,0,255),
                    (255,0,255,255),(0,127,255,255),(127,0,0,255),(0,127,0,255),
                    (0,0,127,255),(127,255,0,255),(127,0,255,255),(0,127,255,255)]

            # draw segment
            for idx, segment in enumerate(ext_segments):
                linePen = pg.mkPen(color=colors[idx],width=4,
                    style=QtCore.Qt.SolidLine)
                plot.plot((segment.pt_A[0],segment.pt_B[0]),
                            (segment.pt_A[1],segment.pt_B[1]), pen=linePen)

            
            # draw corners found
            symbolePen = pg.mkPen(color=(0,0,255,255), width= 2)
            if corners is not None:
                corner_x = [corner.x for corner in corners]
                corner_y = [corner.y for corner in corners]
                plot.plot(corner_x, corner_y, pen=None, symbol='x',
                    symbolPen=symbolePen)

            # # draw table estimation corner
            table_corner = np.array([[0,0],[0,config['TABLE_HEIGHT']],[config['TABLE_WIDTH'],
                                            config['TABLE_HEIGHT']],[config['TABLE_WIDTH'],0]], dtype=float)
            table_corner = table_corner - [position[0], position[1]]
            table_corner = rotatePolygon(table_corner, -orientation+math.pi/2)

            symbolePen = pg.mkPen(color=(255,255,255,255), width= 2)
            plot.plot(table_corner[:,0], table_corner[:,1], 
                    pen=None,  symbol='x', symbolPen=symbolePen, symbolSize=12)


            app.processEvents()

if __name__ == '__main__':
    main()
