import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.morphology import skeletonize
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import BezierFit


# Function that performs chord-length parameterisation. Through this function, points on the line are evenly spaced
# along the length of the line
def line_positioning(points):
    # initialise the length vector
    u = [0.0]

    # add a point to u with distance defined by distance between two consecutive points
    for i in range(1, len(points)):
        u.append(u[i - 1] + np.linalg.norm(points[i] - points[i - 1]))

    # normalise u
    u = np.asarray(u)
    #print(u[-1])
    
    u = u / (u[-1] + 1 )
    

    

    return u


# Function that fits a straight line segment to the given coordinate points. Note that here we need to retain the start
# end points of the line otherwise, we will break the continuity of the line strokes.
# points are the coordinate points
# u is the point positioning as defined from the function line_positioning
# function returns Line_coord - the coordinates of points on the straight line
def fit_line(points, u):
    # initialises the coordinate points for the length of the line
    x = np.zeros((len(u), 1))
    y = np.zeros((len(u), 1))

    # find the line orientation
    dx = points[0, 1] - points[-1, 1]  # difference in x direction
    dy = points[0, 0] - points[-1, 0]  # difference in y direction
    theta = np.arctan2(dy, dx)  # angle of line between points

    # find the end-to-end length of the line
    dist = np.linalg.norm(points[0] - points[-1])

    # interpolate the coordinate values along this line
    for i in range(0, len(u)):
        x[i] = points[0, 1] - dist * u[i] * np.cos(theta)
        y[i] = points[0, 0] - dist * u[i] * np.sin(theta)

    # group the (x, y) pairs into a single variable
    Line_coord = np.c_[y, x]

    return Line_coord


# Function that fits a straight line segment to the given coordinate points. Note that here we need to retain the start
# end points of the curve otherwise, we will break the continuity of the line strokes.
# points are the coordinate points
# pts is the point positioning as defined from the function line_positioning
# left_tangent and right_tangent are the estimated left and right tangents of the curve
# function returns Cine_coord - the coordinates of points on the Bezier curve
def fit_bezier(points, pts, left_tangent, right_tangent):
    # fit Schneider's bezier curve control points
    ctrlPoly_SE = BezierFit.generateBezier_Schneider(points, pts, left_tangent, right_tangent)

    # Apply Newton-Raphson Optimisation
    CLP_Prime = [BezierFit.NewtonRaphsonRootFinder(ctrlPoly_SE, point, u) for point, u in zip(points, pts)]
    ctrlPoly_SE = BezierFit.generateBezier_Schneider(points, CLP_Prime, left_tangent, right_tangent)

    # fit the Bezier curve to these points
    Curve_coord = np.zeros((len(pts), 2))  # initialise the curve array

    # loop through all points along the curve
    for k in range(0, len(pts), 1):
        t = k / len(pts)

        # Bernstein's polynomial at point (t)
        Curve_coord[k, :] = BezierFit.CubicBezierCurveValue(ctrlPoly_SE, t)

    return Curve_coord


# function that determines the error between the fitted curve/line and the actual coordinate points. It returns the
# maximum error value and the first occurrence of this value.
def compute_error(fit, actual):
    x2 = fit[:, 1].reshape((len(fit), 1))
    y2 = fit[:, 0].reshape((len(fit), 1))

    vx = actual[:, 1].reshape((len(x2), 1))
    vy = actual[:, 0].reshape((len(x2), 1))

    D = np.sqrt(((vx - x2) ** 2) + ((vy - y2) ** 2))

    D_max = D.max()

    split_point = D.argmax()

    return D_max, split_point


#######################################################################################################################
# The main function of this file. This function uses split and merge to attempt to fit either a line or a Bezier curve
# to the object profile. The best of the two fits, providing that the error is smaller than the maximum defined error.
# coord_points - the object profile
# MaxErr - the largest acceptable error value
# t_env - the support for the tangent estimation
# Returns the variable Sketch with the list of coordinates that approximate the object profile
def object_profile_fitting(coord_points, MaxErr, t_env):
    # determine the length of the sketch stroke
    n = len(coord_points)

    # the initial split point
    splitPoint = 0
    splitPoint = np.asarray(splitPoint)

    # initialise the vector point containers
    VP1 = coord_points
    VP2 = coord_points

    # initialise the sketch approximation coordinates
    #Sketch = np.zeros((1, 2))
    Sketch = (coord_points[0, :]).reshape(1, 2)
    # initialise the critical point list
    idx = list()
    idx.append(0)

    #while (idx[-1] + splitPoint) < len(coord_points):
    while (len(Sketch)) < len(coord_points) - t_env:

        # get the positions on the curve segment
        CLP = line_positioning(VP1)

        # attempt to fit a line to these points
        Line = fit_line(VP1, CLP)

        # --- attempt to fit a curve to these points

        # estimate the left tangent, taking into consideration the central tangents to ensure continuity over the curve
        if (idx[-1] - t_env) < 0:
            # the first line segment tangent
            LT = (coord_points[t_env] - coord_points[0]) / np.linalg.norm(coord_points[t_env] - coord_points[0])

        else:

            # central tangents
            P = coord_points[idx[-1] - t_env: idx[-1] + t_env]
            LT = (P[-1] - P[0]) / np.linalg.norm(P[-1] - P[0])

        # estimate the right tangent, taking into consideration the central tangents to ensure continuity over the curve
        if ((idx[-1] + splitPoint + t_env) < len(coord_points)) and ((idx[-1] + splitPoint - t_env) > 0):

            # central tangents
            P = coord_points[idx[-1] + splitPoint - t_env: idx[-1] + splitPoint + t_env]
            RT = (P[0] - P[-1]) / np.linalg.norm(P[0] - P[-1])

        else:

            # the last line segment tangent
            RT = (coord_points[-t_env] - coord_points[-2]) / np.linalg.norm(coord_points[-t_env] - coord_points[-2])

        # do the curve fitting
        Curve = fit_bezier(VP1, CLP, LT, RT)

        # ---

        # find the error between the fitted line and the vector points
        DMaxLine, splitPointLine = compute_error(Line, VP1)

        # find the error between the fitted line and the vector points
        DMaxCurve, splitPointCurve = compute_error(Curve, VP1)

        if (DMaxLine > MaxErr) and (
                DMaxCurve > MaxErr):  # error is too large, first half of the vector points needs to be split

            # keep the split point corresponding to the least error
            if DMaxCurve < DMaxLine:
                splitPoint = splitPointCurve
            else:
                splitPoint = splitPointLine

            n = splitPoint

            # split the vector points at the split point
            VP1 = coord_points[idx[-1]: splitPoint + idx[-1], :]
            VP2 = coord_points[(splitPoint + idx[-1]):, :]

        else:

            # add previous end point to index list
            idx.append(n + idx[-1])

            if DMaxCurve < DMaxLine:
                n = splitPointCurve
                splitPoint = splitPointCurve
            else:
                n = splitPointLine
                splitPoint = splitPointLine

            VP1 = VP2

            if DMaxCurve < DMaxLine:

                Sketch = np.concatenate((Sketch, Curve), axis=0)


            else:

                Sketch = np.concatenate((Sketch, Line), axis=0)


    idx[-1] = len(coord_points) - 1

    return Sketch
