import pickle
import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))

    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )

    return graph


def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer
    params = gtsam.LevenbergMarquardtParams()

    # TODO: Perform the optimization and print the result
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print("\nFinal Result:\n{}".format(result))

    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_pose = "a"
    best_landmark = 1
    best_sum = float("inf")
    best_total_sum = float("inf")

    for pose_name, pose_5 in pose_options.items():
        pose_best_sum = float("inf")
        pose_best_landmark = 1
        pose_best_total_sum = float("inf")

        for landmark in [1, 2]:
            graph_copy = pickle.loads(pickle.dumps(graph))
            estimate_copy = pickle.loads(pickle.dumps(initial_estimate))

            graph_copy, estimate_copy = add_pose(graph_copy, estimate_copy, pose_5)
            result = optimize(graph_copy, estimate_copy)

            graph_copy = add_landmark_measurement(graph_copy, result, pose_5, landmark)
            result = optimize(graph_copy, result)

            # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
            marginals = gtsam.Marginals(graph_copy, result)

            # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
            sum_for_selection = marginals.marginalCovariance(L(landmark)).sum()

            sum_of_marginals = (
                marginals.marginalCovariance(L(1)).sum()
                + marginals.marginalCovariance(L(2)).sum()
            )

            if sum_for_selection < pose_best_sum:
                pose_best_sum = sum_for_selection
                pose_best_landmark = landmark
                pose_best_total_sum = sum_of_marginals

        if pose_best_sum < best_sum:
            best_sum = pose_best_sum
            best_pose = pose_name
            best_landmark = pose_best_landmark
            best_total_sum = pose_best_total_sum

    return best_pose, best_landmark, best_total_sum

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = "a"
    best_landmark = 1
    best_sum = float("inf")

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            graph_copy = pickle.loads(pickle.dumps(graph))
            estimate_copy = pickle.loads(pickle.dumps(initial_estimate))

            graph_copy, estimate_copy = add_pose(graph_copy, estimate_copy, pose_5)
            result = optimize(graph_copy, estimate_copy)

            graph_copy = add_landmark_measurement(graph_copy, result, pose_5, landmark)
            result = optimize(graph_copy, result)

            # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
            list_of_errors = []
            list_of_errors.append(graph_copy.error(result))

            # TODO: compute the sum of the errors and return it along with the best pose and landmark
            sum_of_errors = sum(list_of_errors)

            if sum_of_errors < best_sum:
                best_sum = sum_of_errors
                best_pose = pose_name
                best_landmark = landmark

    return best_pose, best_landmark, best_sum