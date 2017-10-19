import numpy as np
import ransac
#from scipy.spatial import KDTree
from collections import defaultdict
import models

#NEIGHBORHOOD_RADIUS = 70
GRID_SIZE = 50

class FeaturesMatcher(object):

    def __init__(self, detector, **kwargs):
        self._detector = detector

        self._params = {}
        # get default values if no value is present in kwargs
        #self._params["num_filtered_percent"] = kwargs.get("num_filtered_percent", 0.25)
        #self._params["filter_rate_cutoff"] = kwargs.get("filter_rate_cutoff", 0.25)
        self._params["ROD_cutoff"] = kwargs.get("ROD_cutoff", 0.92)
        self._params["min_features_num"] = kwargs.get("min_features_num", 40)

        # Parameters for the RANSAC
        self._params["model_index"] = kwargs.get("model_index", 3)
        self._params["iterations"] = kwargs.get("iterations", 5000)
        self._params["max_epsilon"] = kwargs.get("max_epsilon", 30.0)
        self._params["min_inlier_ratio"] = kwargs.get("min_inlier_ratio", 0.01)
        self._params["min_num_inlier"] = kwargs.get("min_num_inliers", 7)
        self._params["max_trust"] = kwargs.get("max_trust", 3)
        self._params["det_delta"] = kwargs.get("det_delta", 0.9)
        self._params["max_stretch"] = kwargs.get("max_stretch", 0.25)

        self._params["use_regularizer"] = True if "use_regularizer" in kwargs.keys() else False
        self._params["regularizer_lambda"] = kwargs.get("regularizer_lambda", 0.1)
        self._params["regularizer_model_index"] = kwargs.get("regularizer_model_index", 1)


    def match(self, features_kps1, features_descs1, features_kps2, features_descs2):
        features_kps2 = np.asarray(features_kps2)

        # because the sections were already pre-aligned, we only match a feature in section1 to its neighborhood in section2 (according to a grid)
        grid = defaultdict(set)

        # build the grid of the feature locations in the second section
        for i, kp2 in enumerate(features_kps2):
            pt_grid = (np.array(kp2.pt) / GRID_SIZE).astype(np.int)
            grid[tuple(pt_grid)].add(i)

        match_points = [[], [], []]
        for kp1, desc1 in zip(features_kps1, features_descs1):
            # For each kp1 find the closest points in section2
            pt_grid = (np.array(kp1.pt) / GRID_SIZE).astype(np.int)
            close_kps2_idxs = set()
            # search in a [-1, -1] -> [1, 1] delta windows (3*3)
            for delta_y in range(-1, 2):
                for delta_x in range(-1, 2):
                    delta = np.array([delta_x, delta_y], dtype=np.int)
                    delta_grid_loc = tuple(pt_grid + delta)
                    if delta_grid_loc in grid.keys():
                        close_kps2_idxs |= grid[delta_grid_loc]

            close_kps2_indices = list(close_kps2_idxs)
            close_descs2 = features_descs2[close_kps2_indices]
            matches = self._detector.match(desc1.reshape(1, len(desc1)), close_descs2)
            if len(matches[0]) == 2:
                if matches[0][0].distance < self._params["ROD_cutoff"] * matches[0][1].distance:
                    match_points[0].append(kp1.pt)
                    match_points[1].append(features_kps2[close_kps2_indices][matches[0][0].trainIdx].pt)
                    match_points[2].append(matches[0][0].distance)

       
            

#         # because the sections were already pre-aligned, we only match a feature in section1 to its neighborhood in section2
#         features_kps2_pts = [kp.pt for kp in features_kps2]
#         kps2_pts_tree = KDTree(features_kps2_pts)
# 
#         match_points = [[], [], []]
#         for kp1, desc1 in zip(features_kps1, features_descs1):
#             # For each kp1 find the closest points in section2
#             close_kps2_indices = kps2_pts_tree.query_ball_point(kp1.pt, NEIGHBORHOOD_RADIUS)
#             close_descs2 = features_descs2[close_kps2_indices]
#             matches = self._detector.match(desc1.reshape(1, len(desc1)), close_descs2)
#             if len(matches[0]) == 2:
#                 if matches[0][0].distance < self._params["ROD_cutoff"] * matches[0][1].distance:
#                     match_points[0].append(kp1.pt)
#                     match_points[1].append(features_kps2[close_kps2_indices][matches[0][0].trainIdx].pt)
#                     match_points[2].append(matches[0][0].distance)

        match_points = (np.array(match_points[0]), np.array(match_points[1]), np.array(match_points[2]))


#         matches = self._detector.match(features_descs1, features_descs2)
# 
#         good_matches = []
#         for m, n in matches:
#             #if (n.distance == 0 and m.distance == 0) or (m.distance / n.distance < actual_params["ROD_cutoff"]):
#             if m.distance < self._params["ROD_cutoff"] * n.distance:
#                 good_matches.append(m)
# 
#         match_points = (
#             np.array([features_kps1[m.queryIdx].pt for m in good_matches]),
#             np.array([features_kps2[m.trainIdx].pt for m in good_matches]),
#             np.array([m.distance for m in good_matches])
#         )

        return match_points

    def match_and_filter(self, features_kps1, features_descs1, features_kps2, features_descs2):
        match_points = self.match(features_kps1, features_descs1, features_kps2, features_descs2)

        model, filtered_matches = ransac.filter_matches(match_points, match_points, self._params['model_index'],
                    self._params['iterations'], self._params['max_epsilon'], self._params['min_inlier_ratio'],
                    self._params['min_num_inlier'], self._params['max_trust'], self._params['det_delta'], self._params['max_stretch'])

        if model is None:
            return None, None

        if self._params["use_regularizer"]:
            regularizer_model, _ = ransac.filter_matches(match_points, match_points, self._params['regularizer_model_index'],
                        self._params['iterations'], self._params['max_epsilon'], self._params['min_inlier_ratio'],
                        self._params['min_num_inlier'], self._params['max_trust'], self._params['det_delta'], self._params['max_stretch'])

            if regularizer_model is None:
                return None, None

            result = model.get_matrix() * (1 - self._params["regularizer_lambda"]) + regularizer_model.get_matrix() * self._params["regularizer_lambda"]
            model = models.AffineModel(result)

        return model, filtered_matches

