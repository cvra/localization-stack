import math
import numpy as np
from itertools import product
from skimage.measure import LineModel, ransac
from skimage.transform import AffineTransform
import scipy.ndimage.interpolation
from hull import convex_hull

RANSAC_RESIDUAL_THRESHOLD = 0.03  # in m
RANSAC_MAX_TRIAL = 100
SEGMENT_RESIDUAL_THRESHOLD = 0.09 # in m
INTERSECTION_THRESHOLD = 0.05     # in rad

class Segment:
    def __init__(self, model, points):
        self.pt_A = None
        self.pt_B = None
        self.model = model
        self.points = points

class OrientedCorner:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

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

def keep_border_points(cloud_pts):
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
                                           < SEGMENT_RESIDUAL_THRESHOLD)

    return cloud_pts[to_keep, :]


def fit_line(cloud_pts, sub_sel):
    ''' Fit a line using RANSAC algorithm on a subset of points and return the
        inlier on the whole set of points.

    Parameters
    ----------
    cloud_pts: (N,2) numpy array of float
    sub_sel: (N) numpy array of boolean

    Returns
    -------
    segment: Segment object containing the line model of the fitted line
    inliers: (N) numpy array of boolean '''

    sub_cloud_pts = cloud_pts[sub_sel,:]

    inliers = np.zeros(cloud_pts.shape[0], dtype=bool)
    if sub_cloud_pts.shape[0] < 2:
        return None, inliers

    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, _ = ransac(sub_cloud_pts, LineModel, min_samples=2,
                             residual_threshold=RANSAC_RESIDUAL_THRESHOLD,
                             max_trials=RANSAC_MAX_TRIAL)

    # find inliers on the whole set
    inliers = (abs(model_robust.residuals(cloud_pts))<RANSAC_RESIDUAL_THRESHOLD)
    segment = Segment(model_robust, cloud_pts[inliers,:])

    return segment, inliers

def find_lines(cloud_pts, nb_lines):
    ''' Find N lines in a cloud of points and return list of model of these line

    Parameters
    ----------
    cloud_pts: (N,2) numpy array of float
    nb_lines: integer

    Returns
    -------
    lines_model: list of (M) Segment object containing the line model and the  
    list of points onto each lines . With M = number of found line. '''

    inliers = np.zeros((cloud_pts.shape[0], nb_lines), dtype=bool)
    lines_model = []

    for i in range(nb_lines):
        # select the points that are not yet onto a fitted line
        sel = np.asarray([any(row) == False for row in inliers[:,range(i)]])

        if len(cloud_pts[sel,:]) > 0:
            model, inliers[:,i] = fit_line(cloud_pts, sel)
            if model is not None:
                lines_model.append(model)

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


def keep_external_segment(segments):
    ''' Return segment that are on the convex hull. i.e. all the segment for 
        which all the other segment lies on the same side.

    Parameters
    ----------
    segments: list of (N) Segment object 

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
                             if abs(value) > SEGMENT_RESIDUAL_THRESHOLD]

        residuals_sgn = np.sign(residuals[ind_none_zero])

        if(len(np.unique(residuals_sgn)) == 1):
            kept_segments.append(segment)

    return kept_segments

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

def extract_corner(ext_segments):
    ''' Return the list corner (perpendicular intersection of two segment) 
        characterized by there position and an angular orientation.

    Parameters
    ----------
    ext_segments: list of (N) Segment object 

    Returns
    -------
    corners: list of (M) OrientedCorner object  '''

    corners = []

    for idx1,segment1 in enumerate(ext_segments[:-1]):
        for idx2,segment2 in enumerate(ext_segments[idx1+1:]):
            d1, theta1 = segment1.model.params
            d2, theta2 = segment2.model.params

            diff = abs(theta1 - theta2);

            if abs(diff - math.pi / 2) < INTERSECTION_THRESHOLD:
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
                v1 = v1 / np.linalg.norm(v1)

                v2 = pt2 - pos
                v2 = v2 / np.linalg.norm(v2)

                corner_dir = (v1+v2) / np.linalg.norm(v1+v2)

                angle_ref = math.cos(math.pi/4)

                corner_angle = (math.atan2(-angle_ref,angle_ref) - 
                                math.atan2(corner_dir[1], corner_dir[0])) 

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

def localize(corners, est_position):
    ''' Localize the robot by matching found corners with the one of the 
        rectangular table at there expected position. This need an estimation of
        the position to match correctly the corner.

    Parameters
    ----------
    corners: list of (N) OrientedCorner object 
    est_position: list of 3 float. 1. position x; 2. position y; 3. orientation
                  in radian
    
    '''
    pos = None
    orientation = None

    table_pos = np.array([[0,0],[0,3],[2,3],[2,0]], dtype=float)
    table_angle = np.array([3*math.pi/2,0,math.pi/2,math.pi],dtype=float)

    table_corner = table_pos - [est_position[0], est_position[1]]
    table_corner = rotatePolygon(table_corner, -est_position[2])

    if corners is not None:

        corner_pos = [np.asarray([elem.x,elem.y]) for elem in corners]
        combi = np.asarray([x for x in product(table_corner,corner_pos)])

        norm = [np.linalg.norm(P1-P2) for P1, P2 in combi[:]]
        norm = np.reshape(norm, (table_corner.shape[0], len(corners)))
        match = np.argmin(norm, axis=0)

        orientation =  [corner.angle for corner in corners] - table_angle[match]
        orientation = (orientation + math.pi) % (2*math.pi) - math.pi

        pos = [table_pos[match[idx]] - 
               rotatePolygon(np.array([corner.x, corner.y]), orientation[idx]).T
               for idx,corner in enumerate(corners)]

    return pos, orientation





