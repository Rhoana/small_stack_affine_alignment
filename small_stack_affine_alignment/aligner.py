#from rh_renderer import models
#from rh_aligner.common import ransac
import sys
import os
import glob
import yaml
import cv2
import numpy as np
from rh_logger.api import logger
import logging
import rh_logger
import time
from detector import FeaturesDetector
from matcher import FeaturesMatcher
import multiprocessing as mp

class StackAligner(object):

    def __init__(self, conf, processes_num=1):
        self._conf = conf

        # Initialize the detector, amtcher and optimizer objects
        detector_params = conf.get('detector_params', {})
        matcher_params = conf.get('matcher_params', {})
        self._detector = FeaturesDetector(conf['detector_type'], **detector_params)
        self._matcher = FeaturesMatcher(self._detector, **matcher_params)

        self._processes_num = processes_num



    @staticmethod
    def read_imgs(folder):
        img_fnames = sorted(glob.glob(os.path.join(folder, '*')))[:10]
        print("Loading {} images from {}.".format(len(img_fnames), folder))
        imgs = [cv2.imread(img_fname, 0) for img_fname in img_fnames]
        return img_fnames, imgs


    @staticmethod
    def load_conf_from_file(conf_fname):
        '''
        Loads a given configuration file from a yaml file
        '''
        print("Using config file: {}.".format(conf_fname))
        with open(conf_fname, 'r') as stream:
            conf = yaml.load(stream)
        return conf


    @staticmethod
    def _compute_l2_distance(pts1, pts2):
        delta = pts1 - pts2
        s = np.sum(delta**2, axis=1)
        return np.sqrt(s)

    @staticmethod
    def _compute_features(detector, img, i):
        result = detector.detect(img)
        logger.report_event("Img {}, found {} features.".format(i, len(result[0])), log_level=logging.INFO)
        return result

    @staticmethod
    def _match_features(features_result1, features_result2, i, j):
        transform_model, filtered_matches = self._matcher.match_and_filter(*features_result1, *features_result2)
        assert(transform_model is not None)
        transform_matrix = transform_model.get_matrix()
        logger.report_event("Imgs {} -> {}, found the following transformations\n{}\nAnd the average displacement: {} px".format(i, j, transform_matrix, np.mean(StackAligner._compute_l2_distance(transform_model.apply(filtered_matches[1]), filtered_matches[0]))), log_level=logging.INFO)
        return transform_matrix
 

    def align_imgs(self, imgs):
        '''
        Receives a stack of images to align and aligns that stack using the first image as an anchor
        '''

        #pool = mp.Pool(processes=processes_num)

        # Compute features
        logger.start_process('align_imgs', 'aligner.py', [len(imgs), self._conf])
        logger.report_event("Computing features...", log_level=logging.INFO)
        st_time = time.time()
        all_features = []
        pool_results = []
        for i, img in enumerate(imgs):
            #res = pool.apply_async(StackAligner._compute_features, (self._detector, img, i))
            #pool_results.append(res)
            all_features.append(self._detector.detect(img))
            logger.report_event("Img {}, found {} features.".format(i, len(all_features[-1][0])), log_level=logging.INFO)
        for res in pool_results:
            all_features.append(res.get())
        logger.report_event("Features computation took {} seconds.".format(time.time() - st_time), log_level=logging.INFO)

        # match features of adjacent images
        logger.report_event("Pair-wise feature mathcing...", log_level=logging.INFO)
        st_time = time.time()
        pairwise_transforms = []
        for i in range(len(imgs) - 1):
            transform_model, filtered_matches = self._matcher.match_and_filter(*all_features[i + 1], *all_features[i])
            assert(transform_model is not None)
            transform_matrix = transform_model.get_matrix()
            pairwise_transforms.append(transform_matrix)
            logger.report_event("Imgs {} -> {}, found the following transformations\n{}\nAnd the average displacement: {} px".format(i, i+1, transform_matrix, np.mean(StackAligner._compute_l2_distance(transform_model.apply(filtered_matches[1]), filtered_matches[0]))), log_level=logging.INFO)
        logger.report_event("Feature matching took {} seconds.".format(time.time() - st_time), log_level=logging.INFO)

        # Compute the per-image transformation (all images will be aligned to the first section)
        logger.report_event("Computing transformations...", log_level=logging.INFO)
        st_time = time.time()
        transforms = []
        cur_transform = np.eye(3)
        transforms.append(cur_transform)

        for pair_transform in pairwise_transforms:
            cur_transform = np.dot(cur_transform, pair_transform)
            transforms.append(cur_transform)
        logger.report_event("Transformations computation took {} seconds.".format(time.time() - st_time), log_level=logging.INFO)

        assert(len(imgs) == len(transforms))

        #pool.close()
        #pool.join()

        logger.end_process('align_imgs ending', rh_logger.ExitCode(0))
        return transforms




    @staticmethod
    def align_img_files(imgs_dir, conf, processes_num):
        # Read the files
        _, imgs = StackAligner.read_imgs(imgs_dir)

        aligner = StackAligner(conf, processes_num)
        return aligner.align_imgs(imgs)


if __name__ == '__main__':
    imgs_dir = '/n/coxfs01/paragt/Adi/R0/images_margin'
    conf_fname = '../conf_example.yaml'
    out_path = './output_imgs'
    processes_num = 8

    conf = StackAligner.load_conf_from_file(conf_fname)
    transforms = StackAligner.align_img_files(imgs_dir, conf, processes_num)

    # Save the transforms to a temp output folder
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print('Writing output to: {}'.format(out_path))
    img_fnames, imgs = StackAligner.read_imgs(imgs_dir)
    for img_fname, img, transform in zip(img_fnames, imgs, transforms):
        # assumption: the output image shape will be the same as the input image
        out_fname = os.path.join(out_path, os.path.basename(img_fname))
        img_transformed = cv2.warpAffine(img, transform[:2,:], (img.shape[1], img.shape[0]), flags=cv2.INTER_AREA)
        cv2.imwrite(out_fname, img_transformed)

