import argparse
import json
from jsmin import jsmin
import numpy as np
import os
import time
import zmqmsgbus
from convex_area_localization import *
from models import PositionModel
from skimage.measure import ransac

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

    x = radius * np.cos(theta + np.pi / 2)
    y = radius * np.sin(theta + np.pi / 2)

    cartesian = np.array([x, y]).T;
    return cartesian;


def update_scan_radius(topic, message, args, config, node):
    global radius

    radius = np.array(message)
    positioning(args, config, node)


def update_scan_theta(topic, message):
    global theta
    theta = np.array(message)


def update_scan_pos(topic, message):
    global datagram_pos
    datagram_pos = get_lidar_position_from_robot(message)


def get_position_using_corners(segments, config, datagram_pos, node, args):
    corners = None
    # only keep external segments
    ext_segments = keep_external_segment(segments=segments, 
                                         segment_residual_threshold=config['SEGMENT_RESIDUAL_THRESHOLD'],
                                         segment_total_residual_threshold=config['SEGMENT_TOTAL_RESIDUAL_THRESHOLD'])

    # extract valide intersections
    corners = extract_corner(ext_segments=ext_segments, 
                             intersection_threshold=config['INTERSECTION_THRESHOLD'])

    # localize the robot using the intersections found
    positions, headings = localize_using_corners(corners=corners, 
                                                 est_position=datagram_pos, 
                                                 config=config)

    if headings is None:
        position = (None, None)
        heading = None
    elif len(headings) > 1:
        positions = np.vstack(positions)
        headings = np.vstack(headings)        

        model_robust, inliers = ransac(data=(positions, headings),
                                       model_class=PositionModel,
                                       min_samples=1,
                                       residual_threshold=1, 
                                       max_trials=10)
        outliers = inliers == False

        model_robust.estimate(positions[inliers], headings[inliers])

        position = model_robust.position
        heading = model_robust.heading
    else:
        position = positions[0].squeeze()
        heading = headings[0]

    if args.print_output:
        os.system('clear')
        print(str(position)+" "+str(heading))

    # Publish data for viewer 
    if args.logs:
        segments_publisher = list()
        for idx, segment in enumerate(ext_segments):
            segments_publisher.append(((segment.pt_A[0], segment.pt_B[0]), (segment.pt_A[1],segment.pt_B[1])))
        node.publish('/lidar_viewer/ext_segments', segments_publisher)

        if corners is not None:
            node.publish('/lidar_viewer/corners', [(corner.x, corner.y) for corner in corners])
        else:
            node.publish('/lidar_viewer/corners', [])

    return position, heading


def get_position_using_features(segments, config, datagram_pos, node, args):
    position = (None, None)
    heading = None

    # extract valide intersections as feature
    features = extract_corner(ext_segments=segments, 
                              intersection_threshold=config['INTERSECTION_THRESHOLD'])

    if features is not None and len(features) > 1:
        # localize the robot using the intersections found
        position, heading = localize_using_landmarks(features=features, 
                                                         est_position=datagram_pos, 
                                                         config=config)

    # Publish data for viewer 
    if args.logs:
        if features is not None:
            node.publish('/lidar_viewer/features', [(feature.x, feature.y) for feature in features])
        else:
            node.publish('/lidar_viewer/features', [])

    return position, heading


def positioning(args, config, node):
    global radius, theta
    global datagram_pos

    position = (None, None)
    heading = None

    # remove points that are too close
    radius_filtered, theta_filtered = remove_off_range_pts(radius=radius, 
                                                           theta=theta, 
                                                           min_distance=config['CLOSEST_POINT'], 
                                                           max_distance=config['FAREST_POINT'])

    # convert polar coordinate in cartesian
    cloud_pts = pol2cart(radius_filtered, theta_filtered)

    # reduce cloud point density
    red_cloud_pts = density_reduction(cloud_pts, config['MAX_DIST_BETWEEN_POINT'], 1)

    # find lines
    lines_model = find_lines(cloud_pts=red_cloud_pts, 
                             nb_lines=config['NB_LINE'], 
                             ransac_residual_threshold=config['RANSAC_RESIDUAL_THRESHOLD'], 
                             ransac_max_trial=config['RANSAC_MAX_TRIAL'])

    # convert lines onto segments
    segments = segment_line(lines_model)

    # localize the robot using the table corners 
    position_crn, heading_crn = get_position_using_corners(segments=segments, 
                                                           config=config, 
                                                           datagram_pos=datagram_pos,
                                                           node=node,
                                                           args=args)

    # # localize the robot using the known features on the table 
    # position_ftr, heading_ftr = get_position_using_features(segments=segments, 
    #                                                         config=config, 
    #                                                         datagram_pos=datagram_pos,
    #                                                         node=node,
    #                                                         args=args)
    
    if position_crn is not None  and heading_crn is not None:
        node.publish('/lidar_debug/position_crn', get_robot_position_from_lidar(position_crn.tolist() + [heading_crn.tolist()]))
    # if position_ftr is not None  and heading_ftr is not None:
    #     node.publish('/lidar_debug/position_ftr', get_robot_position_from_lidar(position_ftr.tolist() + [heading_ftr.tolist()]))

    # if all([all(position_crn), heading_crn, all(position_ftr), heading_ftr]):
    #     position, heading = filter_positions(last_pos=[np.array(datagram_pos[:2]), np.array(datagram_pos[2])],
    #                                          positions=[[position_crn, heading_crn], [position_ftr, heading_ftr]],
    #                                          options=config)
    # elif all([all(position_crn), heading_crn]):
    if all([all(position_crn), heading_crn]):
        position, heading = filter_one_position(last_pos=[np.array(datagram_pos[:2]), np.array(datagram_pos[2])],
                                                position=[position_crn, heading_crn],
                                                options=config)
    # elif all([all(position_ftr), heading_ftr]):
    #     position, heading = filter_one_position(last_pos=[np.array(datagram_pos[:2]), np.array(datagram_pos[2])],
    #                                             position=[position_ftr, heading_ftr],
    #                                             options=config)

    if position is not None and heading is not None:
        node.publish('/lidar/position', get_robot_position_from_lidar(position.tolist() + [heading.tolist()]))

    # Publish data for viewer 
    if args.logs:
        node.publish('/lidar_viewer/cloud_pts', cloud_pts.tolist())
        node.publish('/lidar_viewer/red_cloud_pts', red_cloud_pts.tolist())
        
        segments_publisher = list()
        for idx, segment in enumerate(segments):
            segments_publisher.append(((segment.pt_A[0], segment.pt_B[0]), (segment.pt_A[1],segment.pt_B[1])))
        node.publish('/lidar_viewer/segments', segments_publisher)

def main():
    global radius, theta
    global datagram_pos

    args = parse_args()
    config = json.loads(jsmin(args.config.read()))

    # zmqmsgbus subscribe
    bus = zmqmsgbus.Bus(sub_addr='ipc://ipc/source',
                        pub_addr='ipc://ipc/sink')
    node = zmqmsgbus.Node(bus)

    theta = np.array(node.recv('/lidar/theta'))
    node.register_message_handler('/lidar/theta', update_scan_theta)

    datagram_pos = node.recv('/position')
    node.register_message_handler('/position', update_scan_pos)

    radius = np.array(node.recv('/lidar/radius'))

    node.register_message_handler('/lidar/radius', lambda topic, message: update_scan_radius(topic, message, args, config, node))

    print('receiving')
    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
