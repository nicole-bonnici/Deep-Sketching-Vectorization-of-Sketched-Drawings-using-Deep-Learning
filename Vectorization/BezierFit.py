# Implementation of "Algorithm for Automatically Fitting Digitized Curves" by Philip J. Schneider from "Graphics Gems", Academic Press, 1990

import numpy as np
import matplotlib.pyplot as plt


# CHORD LENGTH PARAMETERISATION
def ChordLengthParameterisaion(points):
    # initialise the length vector
    u = [0.0]

    # add a point to u with distance defined by distance between two consecutive points
    for i in range(1, len(points)):
        u.append(u[i - 1] + np.linalg.norm(points[i] - points[i - 1]))

    # normalise u by total length to a final range of [0, 1]
    u = u / u[-1]

    return u


# FINDS THE CUBIC BEZIER FIT AT POSITION t
def CubicBezierCurveValue(ctrlPoly, t):
    # evaluates Bernstein's polynomial at point (t)
    b = (1.0 - t) ** 3 * ctrlPoly[0] + 3 * (1.0 - t) ** 2 * t * ctrlPoly[1] + 3 * (1.0 - t) * t ** 2 * ctrlPoly[
        2] + t ** 3 * ctrlPoly[3]

    return b


# FINDS THE FIRST DERIVATIVE OF THE BEZIER CURVE AT POSITION t
def CubicBezierFirstDerivative(ctrlPoly, t):
    # evaluates first derivative at point (t)
    b_dev = 3 * (1.0 - t) ** 2 * (ctrlPoly[1] - ctrlPoly[0]) + 6 * (1.0 - t) * t * (
                ctrlPoly[2] - ctrlPoly[1]) + 3 * t ** 2 * (ctrlPoly[3] - ctrlPoly[2])

    return b_dev


# FINDS THE SECOND DERIVATIVE OF THE BEZIER CURVE AT POSITION t
def CubicBezierSecondDerivative(ctrlPoly, t):
    # evaluates first derivative at point (t)
    b_s_dev = 6 * (1.0 - t) * (ctrlPoly[2] - 2 * ctrlPoly[1] + ctrlPoly[0]) + 6 * (t) * (
                ctrlPoly[3] - 2 * ctrlPoly[2] + ctrlPoly[1])

    return b_s_dev


def generateBezier_Schneider(VectorPoints, CLP, LT, RT):
    # Using the initial tangent estimates to find the control points that minimsies dist = || P - Q(t) || (Eqn 1, pg 617). This requries determining matrices C and X
    # whose determinants allows us to obtain the values of alpha_l and alpha_r which can be used to make an intiial estimate of the placement of the inner control
    # points of the Bezier curve.
    #
    # The mathematical derivations for these values are found on pgs 618 - 619 of Graphical Gems
    #
    # LT and RT are the left and right tangents of the curve
    # CLP is the chord length parameterisation

    # set the initial container of the control points
    ctrlPoly = [VectorPoints[0], None, None, VectorPoints[-1]]

    # compute the A's
    A = np.zeros((len(CLP), 2, 2))
    for i, u in enumerate(CLP):
        A[i][0] = LT * 3 * (1 - u) ** 2 * u
        A[i][1] = RT * 3 * (1 - u) * u ** 2

    # Create the C and X matrices
    C = np.zeros((2, 2))
    X = np.zeros(2)

    for i, (point, u) in enumerate(zip(VectorPoints, CLP)):
        C[0][0] += np.dot(A[i][0], A[i][0])
        C[0][1] += np.dot(A[i][0], A[i][1])
        C[1][0] += np.dot(A[i][0], A[i][1])
        C[1][1] += np.dot(A[i][1], A[i][1])

        tmp = point - CubicBezierCurveValue([VectorPoints[0], VectorPoints[0], VectorPoints[-1], VectorPoints[-1]], u)

        X[0] += np.dot(A[i][0], tmp)
        X[1] += np.dot(A[i][1], tmp)

    # Compute the determinants of C and X
    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X = C[0][0] * X[1] - C[1][0] * X[0]
    det_X_C1 = X[0] * C[1][1] - X[1] * C[0][1]

    # Derive alpha values
    alpha_l = 0.0 if det_C0_C1 == 0 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if det_C0_C1 == 0 else det_C0_X / det_C0_C1

    # If any of the alpha_l or alpha_r is a negative value, or 0, subsequent Newton Raphosn optimisation will result in divide by zeros. In such cases, the heuristic
    # placement of the intermediate control points i.e. placing them at the 1/3 points will be adopted.

    dist = np.linalg.norm(VectorPoints[0] - VectorPoints[-1]) / 3.0

    epsilon = 1.0e-6 * dist

    if alpha_l < epsilon or alpha_r < epsilon:

        # using the 1/3 placement heuristing
        ctrlPoly[1] = ctrlPoly[0] + LT * dist
        ctrlPoly[2] = ctrlPoly[3] + RT * dist

    else:
        # using alpha_1 and alpha_2 to place the intermediate control points
        ctrlPoly[1] = ctrlPoly[0] + LT * alpha_l
        ctrlPoly[2] = ctrlPoly[3] + RT * alpha_r

    ctrlPoly = np.asarray(ctrlPoly)

    return ctrlPoly


def generateBezier_Heuristic(VectorPoints, LT, RT):
    # Finds a heurisitc estimation of the control points using the 1/3 divisions of the curve length as placement positions for intermediary points
    # LT and RT are the left and right tangent estimates

    # find the point which is at the 1/3 point of the length of the curve
    dist = np.linalg.norm(VectorPoints[0] - VectorPoints[-1]) / 3.0

    # set the intermediary control points
    ctrlPoly = np.asarray(
        [VectorPoints[0], VectorPoints[0] + LT * dist, VectorPoints[-1] + RT * dist, VectorPoints[-1]])

    return ctrlPoly


def NewtonRaphsonRootFinder(ctrlPoly, VectorPoints, u):
    # difference between cubic bezier fit and actual vector points
    d = CubicBezierCurveValue(ctrlPoly, u) - VectorPoints

    # first derivative at u
    b1 = CubicBezierFirstDerivative(ctrlPoly, u)

    # second derivative at u
    b2 = CubicBezierSecondDerivative(ctrlPoly, u)

    numerator = (d * b1).sum()
    denominator = (b1 ** 2 + d * b2).sum()

    if denominator == 0.0:
        return u
    else:
        return u - numerator / denominator
