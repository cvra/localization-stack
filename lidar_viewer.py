from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import argparse
import json
import math
from jsmin import jsmin
import zmqmsgbus
import numpy as np
from convex_area_localization import rotatePolygon

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('config', type=argparse.FileType('r'), help='File where the configuration is stored.')

    return parser.parse_args()


def update_position(topic, message):
    global position, orientation
    msg = np.asarray(message)
    position = msg[:2]
    orientation = msg[2]


def update_cloud_pts(topic, message):
    global cloud_pts, cloud_pts_plot
    cloud_pts = np.asarray(message)


def update_red_cloud_pts(topic, message):
    global red_cloud_pts
    red_cloud_pts = np.asarray(message)


def update_segments(topic, message):
    global segments
    segments = np.asarray(message)


def update_corners(topic, message):
    global corners
    corners = np.asarray(message)


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


def main():
    global position, orientation
    global cloud_pts, cloud_pts_plot
    global red_cloud_pts
    global convexHulls
    global segments
    global corners

    args = parse_args()
    config = json.loads(jsmin(args.config.read()))

    # zmqmsgbus subscribe
    bus = zmqmsgbus.Bus(sub_addr='ipc://ipc/source',
                        pub_addr='ipc://ipc/sink')
    node = zmqmsgbus.Node(bus)

    position = node.recv('/lidar/position')
    node.register_message_handler('/lidar/position', update_position)
    cloud_pts = node.recv('/lidar_viewer/cloud_pts')
    node.register_message_handler('/lidar_viewer/cloud_pts', update_cloud_pts)
    red_cloud_pts = node.recv('/lidar_viewer/red_cloud_pts')
    node.register_message_handler('/lidar_viewer/red_cloud_pts', update_red_cloud_pts)
    segments = node.recv('/lidar_viewer/segments')
    node.register_message_handler('/lidar_viewer/segments', update_segments)
    corners = node.recv('/lidar_viewer/corners')
    node.register_message_handler('/lidar_viewer/corners', update_corners)

    # PyQtGraph stuff
    app = QtGui.QApplication([])
    pg.setConfigOptions(antialias=False)
    plot = pg.plot(title='Lidar Polar Plot')
    plot.resize(600,600)
    plot.setAspectLocked()

    while 1:
        plot.clear()
        PlotPolar(plot)

        cloud_size = red_cloud_pts.shape[0]
        plot.plot(red_cloud_pts[np.arange(0,cloud_size,4),0],red_cloud_pts[np.arange(0,cloud_size,4),1],pen=None,symbol='o', symbolPen=None,symbolSize=7,symbolBrush=(255,255,255,50))


        linePen = pg.mkPen(color=(200,200,200,200),width=2,
            style=QtCore.Qt.DotLine)

        colors=[(255,0,0,180),(0,255,0,180),(0,0,255,180),(255,255,0,180),
                (255,0,255,180),(0,127,255,180),(127,0,0,180),(0,127,0,180),
                (0,0,127,180),(127,255,0,180),(127,0,255,180),(0,127,255,180)]

        # draw segment
        try:
            for idx in range(segments.shape[0]):
                linePen = pg.mkPen(color=colors[idx],width=4,
                    style=QtCore.Qt.SolidLine)
                a = plot.plot((segments[idx][0][0], segments[idx][0][1]),
                          (segments[idx][1][0], segments[idx][1][1]), pen=linePen)
        except:
            pass

        # draw corners found
        symbolePen = pg.mkPen(color=(255,255,0,255), width=4)
        print(corners.shape)
        if corners.shape[0] > 0:
            plot.plot(corners[:,0], corners[:,1], pen=None, symbol='x',
                symbolPen=symbolePen)

        # draw table estimation corner
        table_corner = np.array([[0,0],[0,config['TABLE_HEIGHT']],[config['TABLE_WIDTH'],
                                        config['TABLE_HEIGHT']],[config['TABLE_WIDTH'],0]], dtype=float)
        table_corner = table_corner - [position[0], position[1]]
        table_corner = rotatePolygon(table_corner, -orientation+math.pi/2)

        symbolePen = pg.mkPen(color=(255,255,255,255), width= 2)
        plot.plot(table_corner[:,0], table_corner[:,1], 
                pen=None,  symbol='x', symbolPen=symbolePen, symbolSize=12)

        app.processEvents()

    QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    main()