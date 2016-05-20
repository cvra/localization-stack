import argparse
import json
from jsmin import jsmin
import numpy as np
import os

import zmqmsgbus
from convex_area_localization import *

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('config', type=argparse.FileType('r'), help='File where the configuration is stored.')

    parser.add_argument('--print', dest='print_output', help='Print estimated postion and orientation in the shell.', action="store_true")

    parser.add_argument('--logs', help='Logs and publish intermediate computations.', action="store_true")

    return parser.parse_args()


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
        lines_model = find_lines(cloud_pts=red_cloud_pts, 
                                 nb_lines=config['NB_LINE'], 
                                 ransac_residual_threshold=config['RANSAC_RESIDUAL_THRESHOLD'], 
                                 ransac_max_trial=config['RANSAC_MAX_TRIAL'])

        # convert lines onto segments
        segments = segment_line(lines_model)

        ext_segments = segments
        # only keep external segments
        # ext_segments = keep_external_segment(segments=segments, 
        #                                      segment_residual_threshold=config['SEGMENT_RESIDUAL_THRESHOLD'],
        #                                      segment_total_residual_threshold=config['SEGMENT_TOTAL_RESIDUAL_THRESHOLD'])

        # extract valide intersections
        corners = extract_corner(ext_segments=ext_segments, 
                                 intersection_threshold=config['INTERSECTION_THRESHOLD'])

        # localize the robot using the intersections found
        positions, orientations = localize(corners=corners, 
                                           est_position=datagram_pos, 
                                           table_width=config['TABLE_WIDTH'], 
                                           table_height=config['TABLE_HEIGHT'])

        position = np.array([0,0])
        orientation = np.array([0])

        if positions is not None and orientations is not None:
            position = np.mean(positions, axis=0).squeeze()
            orientation = np.mean(orientations)

            if args.print_output:
                os.system('clear')
                print(str(position)+" "+str(orientation))

            node.publish('/lidar/position', position.tolist() + [orientation])

        # Publish data for viewer 
        if args.logs:
            node.publish('/lidar_viewer/cloud_pts', cloud_pts.tolist())
            node.publish('/lidar_viewer/red_cloud_pts', red_cloud_pts.tolist())
            
            segments_publisher = list()
            for idx, segment in enumerate(ext_segments):
                segments_publisher.append(((segment.pt_A[0], segment.pt_B[0]), (segment.pt_A[1],segment.pt_B[1])))
            node.publish('/lidar_viewer/segments', segments_publisher)

            if corners is not None:
                node.publish('/lidar_viewer/corners', [(corner.x, corner.y) for corner in corners])
            else:
                node.publish('/lidar_viewer/corners', [])
if __name__ == '__main__':
    main()
