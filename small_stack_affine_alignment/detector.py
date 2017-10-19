import cv2
from enum import Enum
import numpy as np

class FeaturesDetector(object):

    class Type(Enum):
        SIFT = 1
        ORB = 2
        SURF = 3
        BRISK = 4
        AKAZE = 5

    def __init__(self, detector_type_name, **kwargs):
        detector_type = FeaturesDetector.Type[detector_type_name]
        if detector_type == FeaturesDetector.Type.SIFT:
            self._detector = cv2.xfeatures2d.SIFT_create(**kwargs)
            self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif detector_type == FeaturesDetector.Type.ORB:
            cv2.ocl.setUseOpenCL(False) # Avoiding a bug in OpenCV 3.1
            self._detector = cv2.ORB_create(**kwargs)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif detector_type == FeaturesDetector.Type.SURF:
            self._detector = cv2.xfeatures2d.SURF_create(**kwargs)
            self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif detector_type == FeaturesDetector.Type.BRISK:
            self._detector = cv2.BRISK_create(**kwargs)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif detector_type == FeaturesDetector.Type.AKAZE:
            self._detector = cv2.AKAZE_create(**kwargs)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            raise("Unknown feature detector algorithm given")


    def detect(self, img):
        return self._detector.detectAndCompute(img, None)

    def match(self, features_descs1, features_descs2):
        return self._matcher.knnMatch(features_descs1, features_descs2, k=2)
