import math
import numpy as np
from itertools import product
from skimage.measure import LineModel, ransac
from hull import convex_hull
from models import TransformationModel, mean_angle, diff_angle, position_residuals

class Segment:
    def __init__(self, model, points):
        self.pt_A = None
        self.pt_B = None
        self.model = model
        self.points = points

    def length(self):
        length = 0
        pts_A = np.asarray(self.pt_A)
        pts_B = np.asarray(self.pt_B)
        if self.pt_A is not None and self.pt_A is not None: 
            length = np.linalg.norm(pts_A - pts_B)

        return length


class OrientedCorner:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        

def get_robot_position_from_lidar(lidar):
    lidar_heading = lidar[2]
    robot_position = np.array(lidar[0:2]) - np.array([math.cos(lidar_heading), math.sin(lidar_heading)]) * 0.074
    return [robot_position[0], robot_position[1], lidar_heading]


def get_lidar_position_from_robot(robot):
    robot_heading = robot[2]
    lidar_position = np.array(robot[0:2]) + np.array([math.cos(robot_heading), math.sin(robot_heading)]) * 0.074
    return [lidar_position[0], lidar_position[1], robot_heading]


def remove_off_range_pts(radius, theta, min_distance, max_distance):
    to_keep = np.array([r >= min_distance and r <= max_distance for r in radius])
    return radius[to_keep], theta[to_keep]


def density_reduction(cloud_pts, min_dist_treshold, k):
    ''' Received a cloud of point, and reduce the spatial density by removing
        consecutive points that are closer than a threshold. Each point is
        compared to the k-th following point.

    Parameters
    ----------
    cloud_pts: (N,2) numpy array
    min_dist_treshold: float
    k: integer

    Returns
    -------
    filtered_points: (N,2) numpy array '''

    length = cloud_pts.shape[0]
    ind = np.ones(length, dtype=bool)

    threshold_square = math.pow(min_dist_treshold,2);

    for i in range(length-k):
        if ind[i] == True:
            j = i+k
            X = cloud_pts[i,0]-cloud_pts[j,0]
            Y = cloud_pts[i,1]-cloud_pts[j,1]

            while X*X + Y*Y < threshold_square:
                ind[j] = False
                j += 1
                if j >= length:
                    break
                Y = cloud_pts[i,1]-cloud_pts[j,1]

    filtered_points = cloud_pts[ind,:];

    return filtered_points;

def keep_border_points(cloud_pts, convex_hull_threshold):
    ''' Return the points that are localized onto the convex hull of the cloud
        of input points.

    Input and output are (N,2) numpy array '''

    convexHulls = convex_hull(cloud_pts.tolist())
    convexHulls[:+1][:] = convexHulls[0][:]

    nsegments = len(convexHulls)

    to_keep = np.zeros(len(cloud_pts), dtype=bool)

    for P1, P2 in zip(convexHulls[:-1][:], convexHulls[1:][:]):
        data = np.asarray([P1,P2])

        model = LineModel()
        model.estimate(data)

        to_keep = np.logical_or(to_keep, abs(model.residuals(cloud_pts)) 
                                           < convex_hull_threshold)

    return cloud_pts[to_keep, :]

def fit_line(cloud_pts, ransac_residual_threshold, ransac_max_trial):
    ''' Fit a line using RANSAC algorithm on a set of points and return the
        inlier on the whole set of points.

    Parameters
    ----------
    cloud_pts: (N,2) numpy array of float
    ransac_residual_threshold: maximum residual threshold for RANSAC
    ransac_max_trial: maximum number of trials that RANSAC must perform before
                      stopping to search.

    Returns
    -------
    segment: Segment object containing the line model of the fitted line
    inliers: (N) numpy array of boolean '''

    if cloud_pts.shape[0] < 2:
        return None

    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(cloud_pts, LineModel, min_samples=2,
                             residual_threshold=ransac_residual_threshold,
                             max_trials=ransac_max_trial)
    model_robust.estimate(cloud_pts[inliers])


    inliers_pts = cloud_pts[inliers,:]
    segment = Segment(model_robust, inliers_pts)

    return segment


def find_lines(cloud_pts, nb_lines, ransac_residual_threshold, ransac_max_trial):
    ''' Find N lines in a cloud of points and return list of model of these line

    Parameters
    ----------
    cloud_pts: (N,2) numpy array of float
    nb_lines: integer
    ransac_residual_threshold: maximum residual threshold for RANSAC
    ransac_max_trial: maximum number of trials that RANSAC must perform before
                      stopping to search.

    Returns
    -------
    lines_model: list of (M) Segment object containing the line model and the  
    list of points onto each lines . With M = number of found line. '''

    lines_model = []
    sub_cloud_pts = cloud_pts
    for i in range(nb_lines):
        if len(sub_cloud_pts) > 0:
            model = fit_line(cloud_pts=sub_cloud_pts, 
                             ransac_residual_threshold=ransac_residual_threshold,
                             ransac_max_trial=ransac_max_trial)

            if model is not None:
                # if len(model.points) > MIN_NB_POINTS_PER_SEGMENT:
                #     lines_model.append(model)
                lines_model.append(model)

                set_sub = set(tuple(x) for x in sub_cloud_pts)
                set_model = set(tuple(x) for x in model.points)

                sub_cloud_pts = np.array([x for x in set_sub - set_model])

    return lines_model

def segment_line(lines_model):
    ''' Project all inliers points of each line to find the extremas of these 
        lines and update the Segment object with thhese extremas.

    Parameters
    ----------
    lines_model: list of (N) Segment object '''

    nb_line = len(lines_model)

    segment = []

    for line in lines_model:

        proj_pts_x = line.points[:,0] + (line.model.residuals(line.points) * 
                                         math.cos(line.model.params[1]))
        proj_pts_y = line.points[:,1] + (line.model.residuals(line.points) * 
                                         math.sin(line.model.params[1]))

        p1 = np.argmin(proj_pts_x)
        p2 = np.argmax(proj_pts_x)

        if p1 == p2:
            p1 = np.argmin(proj_pts_y)
            p2 = np.argmax(proj_pts_y)

        line.pt_A = [proj_pts_x[p1], proj_pts_y[p1]]
        line.pt_B = [proj_pts_x[p2], proj_pts_y[p2]]

    return lines_model

def keep_external_segment(segments, segment_residual_threshold, segment_total_residual_threshold):
    ''' Return segment that are on the convex hull. i.e. all the segment for 
        which all the other segment lies on the same side.

    Parameters
    ----------
    segments: list of (N) Segment object 
    segment_residual_threshold: Maximum distance around convex hull 
    segment_total_residual_threshold: 

    Returns
    -------
    kept_segments: list of (M) Segment object '''

    kept_segments = []

    ptsA = [s.pt_A for s in segments]
    ptsB = [s.pt_B for s in segments]

    pts = np.concatenate((ptsA, ptsB), axis=0)

    for segment in segments:
        residuals = segment.model.residuals(pts)
        ind_none_zero = [idx for idx,value in enumerate(residuals)   
                             if abs(value) > segment_residual_threshold]

        residuals_sgn = np.sign(residuals[ind_none_zero])

        if(len(np.unique(residuals_sgn)) == 1):
            kept_segments.append(segment)

    kept_idx = np.ones(len(kept_segments), dtype=bool)

    for idx1,segment1 in enumerate(kept_segments[:-1]):
        for idx_rel,segment2 in enumerate(kept_segments[idx1+1:]):
            if kept_idx[idx1] == True:
                idx2 = idx1 + idx_rel + 1

                residuals = segment1.model.residuals(np.asarray([segment2.pt_A, segment2.pt_B]))

                if max(abs(residuals)) < segment_total_residual_threshold:
                    if(segment1.length() > segment2.length()):
                        kept_idx[idx2] = False
                    else:
                        kept_idx[idx1] = False

    ext_segment = [s for idx, s in enumerate(kept_segments) if kept_idx[idx] == True]
    return ext_segment

def segment_intersection(segment1, segment2):
    ''' Return the intersection point of the segment

    Parameters
    ----------
    segments1, segments2: Segment object that containe a LineModel object

    Returns
    -------
    x, y: float '''

    d1, theta1 = segment1.model.params
    d2, theta2 = segment2.model.params

    x = (d2*math.sin(theta1) - d1*math.sin(theta2)) / math.sin(theta1 - theta2)
    y = (d2*math.cos(theta1) - d1*math.cos(theta2)) / math.sin(theta2 - theta1)

    return x, y

def extract_corner(ext_segments, intersection_threshold):
    ''' Return the list corner (perpendicular intersection of two segment) 
        characterized by there position and an angular orientation.

    Parameters
    ----------
    ext_segments: list of (N) Segment object 
    intersection_threshold:

    Returns
    -------
    corners: list of (M) OrientedCorner object  '''

    corners = []

    for idx1,segment1 in enumerate(ext_segments[:-1]):
        for idx2,segment2 in enumerate(ext_segments[idx1+1:]):
            d1, theta1 = segment1.model.params
            d2, theta2 = segment2.model.params

            diff = abs(theta1 - theta2);

            if abs(diff - math.pi / 2) < intersection_threshold:
                x, y = segment_intersection(segment1, segment2)

                pos = np.array([x,y])
                normA1 = np.linalg.norm(segment1.pt_A-pos)
                normB1 = np.linalg.norm(segment1.pt_B-pos)
                normA2 = np.linalg.norm(segment2.pt_A-pos)
                normB2 = np.linalg.norm(segment2.pt_B-pos)

                if normA1 > normB1:
                    pt1 = segment1.pt_A
                else:
                    pt1 = segment1.pt_B

                if normA2 > normB2:
                    pt2 = segment2.pt_A
                else:
                    pt2 = segment2.pt_B

                v1 = pt1 - pos
                v2 = pt2 - pos
                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)

                corner_dir = (v1+v2)

                corner_angle = math.atan2(corner_dir[1], corner_dir[0]) - math.atan2(1,1)
                corner_angle = corner_angle % (2*math.pi)

                corner = OrientedCorner(x, y, corner_angle)
                corners.append(corner)

    if len(corners) == 0:
        return None

    return corners

def rotatePolygon(polygon,theta):
    '''Rotate a 2-d polygon in a 2-d plane

    Parameters
    ----------
    polygon: (N,2) numpy array of float
    theta: angle in radian

    Returns
    -------
    rotated_polygon: (N,2) numpy array of float '''

    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    rotMat = np.asmatrix([[cosTheta, -sinTheta],[sinTheta, cosTheta]])
    rotated_polygon = np.asarray(np.dot(rotMat, polygon.T).T)

    return rotated_polygon


def localize_using_corners(corners, est_position, config):
    ''' Localize the robot by matching found corners with the one of the 
        rectangular table at there expected position. This need an estimation of
        the position to match correctly the corner.

    Parameters
    ----------
    corners: list of (N) OrientedCorner object 
    est_position: list of 3 float. 1. position x; 2. position y; 3. orientation
                  in radian
    
    '''
    pos = (None, None)
    orientation = None

    table_pos = np.array([[0,0],
                          [config['TABLE_WIDTH'],0],
                          [config['TABLE_WIDTH'],config['TABLE_HEIGHT']],
                          [0,config['TABLE_HEIGHT']]], 
                         dtype=float)
    table_angle = np.array([0,math.pi/2,math.pi,3*math.pi/2],dtype=float)

    table_corner = table_pos - est_position[0:2]
    table_corner = rotatePolygon(table_corner, -est_position[2])

    if corners is not None:
        corner_pos = np.asarray([[corner.x,corner.y] for corner in corners])
        corner_pos = rotatePolygon(corner_pos, -math.pi/2)

        combi = np.asarray([x for x in product(table_corner,corner_pos)])

        norm = [np.linalg.norm(P1-P2) for P1, P2 in combi[:]]
        norm = np.reshape(norm, (table_corner.shape[0], len(corners)))

        match = np.argmin(norm, axis=0)

        orientation = table_angle[match] - [corner.angle-math.pi/2 for corner in corners]
        orientation = (orientation + math.pi) % (2*math.pi) - math.pi

        pos = list()
        for idx,corner in enumerate(corners):
            rotated_corner = rotatePolygon(np.array([corner.x, corner.y]), orientation[idx]-math.pi/2).T
            pos.append(table_pos[match[idx]] - rotated_corner)

    return pos, orientation


def pair_points(ref_points, points):
    paired_list = list()

    for point in points:
        dist = ref_points - point
        dist = dist * dist
        dist = np.sum(dist, axis=1)
        paired_list.append(np.argmin(dist))

    return paired_list


def localize_using_landmarks(features, est_position, config):
    ''' Localize the robot by finding the transformation model (rotation + translation)
        that match the best the set of feature found to the known one on the table.
        This need an estimation of the position.

    Parameters
    ----------
    features: list of (N) OrientedCorner object 
    est_position: list of 3 float. 1. position x; 2. position y; 3. heading
                  in radian
    
    '''
    position = (None, None)
    heading = None

    if features is  None:
        return (None, None), None

    features_list = np.array([(feature.x, feature.y) for feature in features])

    table_landmarks = np.array([[0,0],
                                [config['TABLE_WIDTH'],0],
                                [config['TABLE_WIDTH'],config['TABLE_HEIGHT']],
                                [0,config['TABLE_HEIGHT']]],
                               dtype=float)

    if est_position[1] < config['TABLE_WIDTH'] / 2:
        table_landmarks = np.vstack((table_landmarks, np.array([[0,config['TABLE_HEIGHT']/2 - config['CENTER_OBSTACLE_HALF_WIDTH']], 
                                                                [config['TABLE_WIDTH'],config['TABLE_HEIGHT']/2 - config['CENTER_OBSTACLE_HALF_WIDTH']]])))
    else:
        table_landmarks = np.vstack((table_landmarks, np.array([[0,config['TABLE_HEIGHT']/2 + config['CENTER_OBSTACLE_HALF_WIDTH']], 
                                                                [config['TABLE_WIDTH'],config['TABLE_HEIGHT']/2 + config['CENTER_OBSTACLE_HALF_WIDTH']]])))


    table_rel_landmarks = table_landmarks - est_position[0:2]
    table_rel_landmarks = rotatePolygon(table_rel_landmarks, -est_position[2]+np.pi/2)

    pair_idx = pair_points(table_rel_landmarks, features_list)
    src = table_rel_landmarks[pair_idx]
    dst = features_list

    model_robust, inliers = ransac((src, dst), TransformationModel, min_samples=2,
                                   residual_threshold=config['MAX_RESIDUAL_LANDMARKS'], max_trials=100,
                                   is_data_valid=lambda cls, data: TransformationModel.is_data_valid(cls, data, config['MIN_DISTANCE_VALID_LANDMARKS']))
    outliers = inliers == False

    if model_robust is not None and any(inliers):
        model_robust.estimate(src[inliers], dst[inliers])

        heading = np.array(-model_robust.rotation + est_position[2])
        position = np.array(est_position[0:2]).T + rotatePolygon(model_robust.translation, est_position[2]+np.pi/2).squeeze()

    return position, heading


def are_pos_similar(position1, position2, options):
    err_positions, err_heading = position_residuals(position1=position1.copy(), 
                                                    position2=position2.copy())

    if err_positions > options['MAX_ERR_POSITION']:
        return False
    elif err_heading > options['MAX_ERR_HEADING']:
        return False

    return True


def distance_between_pos(position1, position2, options):
    err_positions, err_heading = position_residuals(position1=position1.copy(), 
                                                    position2=position2.copy())
    return err_positions * options['POS_ERR_FACTOR'] + err_heading * options['HEADING_ERR_FACTOR']


def filter_positions(last_pos, positions, options):
    distance = list()
    distance.append(distance_between_pos(last_pos, positions[0], options=options))
    distance.append(distance_between_pos(last_pos, positions[1], options=options))

    if are_pos_similar(positions[0], positions[1], options=options):
        position = (positions[0][0]+positions[1][0]) / 2
        heading = mean_angle([positions[0][1], positions[1][1]])
        position = np.squeeze(position)
        return position, heading
    else:
        idx_min = np.argmin(distance)
        if distance[idx_min] < options['MAX_POS_DISTANCE']:
            return positions[idx_min]
        else:
            return None, None


def filter_one_position(last_pos, position, options):
    distance = distance_between_pos(last_pos, position, options=options)

    if distance < options['MAX_POS_DISTANCE']:
        return position
    else:
        return None, None
