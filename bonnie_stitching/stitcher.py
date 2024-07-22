import glob
import os
import yaml
import cv2
import numpy as np

from stitching.blender import Blender
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_estimator import CameraEstimator
from stitching.camera_wave_corrector import WaveCorrector
from stitching.cropper import Cropper
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.images import Images
from stitching.seam_finder import SeamFinder
from stitching.warper import Warper


class Stitcher:
    # def stitch(self, frame_number: int, computeHomography: bool = True) -> np.ndarray:
    #     frame_path_like = os.path.join(
    #         "frames", "rectified", f"frame{frame_number}*.png"
    #     )
    #     frame_imgs = glob.glob(frame_path_like)
    #     imgs = Images.of(frame_imgs)

    #     # For the first frame to stitch, compute the homography matrix and save it
    #     if computeHomography:
    #         # Resize the images to medium (and later to low) resolution
    #         medium_imgs = list(imgs.resize(Images.Resolution.MEDIUM))

    #         # Find features
    #         # finder = FeatureDetector(detector="orb", nfeatures=500)
    #         finder = FeatureDetector(detector="orb", nfeatures=10000)
    #         features = [finder.detect_features(img) for img in medium_imgs]

    #         # Match the features of the pairwise images
    #         matcher = FeatureMatcher()
    #         matches = matcher.match_features(features)

    #         # Calibrate cameras which can be used to warp the images
    #         camera_estimator = CameraEstimator()
    #         camera_adjuster = CameraAdjuster()
    #         wave_corrector = WaveCorrector()

    #         cameras = camera_estimator.estimate(features, matches)
    #         cameras = camera_adjuster.adjust(features, matches, cameras)
    #         cameras = wave_corrector.correct(cameras)

    #         # Save camera calibration parameters
    #         self.save_cameras(cameras)

    #         # Warp the images into the final plane
    #         panorama = self.warping_blending(imgs, cameras)
    #     else:
    #         # Get camera calibration parameters
    #         cameras = self.load_cameras()
    #         panorama = self.warping_blending(imgs, cameras)

    #     return panorama

    def stitch(self, frame_number: int, computeHomography: bool = True) -> np.ndarray:
        frame_path_like = os.path.join("frames", "rectified", f"frame{frame_number}*.png")
        frame_imgs = glob.glob(frame_path_like)
        imgs = Images.of(frame_imgs)

        if computeHomography:
            medium_imgs = list(imgs.resize(Images.Resolution.MEDIUM))

            finder = FeatureDetector(detector="orb", nfeatures=10000)
            features = [finder.detect_features(img) for img in medium_imgs]
            
            if any(not hasattr(f, 'keypoints') or len(f.keypoints) == 0 for f in features):
                raise ValueError("No features detected in one or more images")

            matcher = FeatureMatcher()
            matches = matcher.match_features(features)
            
            if len(matches) == 0:
                raise ValueError("No matches found between images")

            camera_estimator = CameraEstimator()
            camera_adjuster = CameraAdjuster()
            wave_corrector = WaveCorrector()

            cameras = camera_estimator.estimate(features, matches)
            cameras = camera_adjuster.adjust(features, matches, cameras)
            cameras = wave_corrector.correct(cameras)

            self.save_cameras(cameras)
        else:
            cameras = self.load_cameras()

        return self.warping_blending(imgs, cameras)


    def save_cameras(self, cameras: list[cv2.detail.CameraParams]):
        yaml_data = {}

        for i, camera in enumerate(cameras):
            data = {
                "focal": camera.focal,
                "aspect": camera.aspect,
                "ppx": camera.ppx,
                "ppy": camera.ppy,
                "R": camera.R.tolist(),
                "t": camera.t.tolist(),
            }
            yaml_data[f"camera{i+1}"] = data

        with open("cameras.yaml", "w") as file:
            # To save the data for better human readability
            # use the default_flow_style=None parameter and sort_keys=False
            yaml.dump(yaml_data, file, default_flow_style=None, sort_keys=False)

    def load_cameras(self) -> list[cv2.detail.CameraParams]:
        cameras = []

        with open("cameras.yaml", "r") as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)

        for camera_name, camera_values in yaml_data.items():
            camera = cv2.detail.CameraParams()

            camera.focal = camera_values["focal"]
            camera.aspect = camera_values["aspect"]
            camera.ppx = camera_values["ppx"]
            camera.ppy = camera_values["ppy"]
            camera.R = np.array(camera_values["R"])
            camera.t = np.array(camera_values["t"])

            cameras.append(camera)

        return cameras

    def warping_blending(self, imgs: Images, cameras: list[str]) -> np.ndarray:
        low_imgs = list(imgs.resize(Images.Resolution.LOW))
        final_imgs = list(imgs.resize(Images.Resolution.FINAL))

        # Select the warper
        warper = Warper(warper_type="paniniA1.5B1")
        # Set the the medium focal length of the cameras as scale
        warper.set_scale(cameras)

        # Warp low resolution images
        low_sizes = []
        for img in low_imgs:
            low_sizes.append(imgs.get_image_size(img))
        camera_aspect = imgs.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)

        warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
        warped_low_masks = list(
            warper.create_and_warp_masks(low_sizes, cameras, camera_aspect)
        )
        low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)

        # Warp final resolution images
        final_sizes = []
        for img in final_imgs:
            final_sizes.append(imgs.get_image_size(img))
        camera_aspect = imgs.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.FINAL
        )

        warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
        warped_final_masks = list(
            warper.create_and_warp_masks(final_sizes, cameras, camera_aspect)
        )
        final_corners, final_sizes = warper.warp_rois(
            final_sizes, cameras, camera_aspect
        )

        # Estimate the largest joint interior rectangle and crop the single images accordingly
        cropper = Cropper()
        mask = cropper.estimate_panorama_mask(
            warped_low_imgs, warped_low_masks, low_corners, low_sizes
        )
        lir = cropper.estimate_largest_interior_rectangle(mask)
        low_corners = cropper.get_zero_center_corners(low_corners)
        rectangles = cropper.get_rectangles(low_corners, low_sizes)
        overlap = cropper.get_overlap(rectangles[1], lir)
        intersection = cropper.get_intersection(rectangles[1], overlap)

        cropper.prepare(warped_low_imgs, warped_low_masks, low_corners, low_sizes)

        cropped_low_masks = list(cropper.crop_images(warped_low_masks))
        cropped_low_imgs = list(cropper.crop_images(warped_low_imgs))
        low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

        lir_aspect = imgs.get_ratio(Images.Resolution.LOW, Images.Resolution.FINAL)
        cropped_final_masks = list(cropper.crop_images(warped_final_masks, lir_aspect))
        cropped_final_imgs = list(cropper.crop_images(warped_final_imgs, lir_aspect))
        final_corners, final_sizes = cropper.crop_rois(
            final_corners, final_sizes, lir_aspect
        )

        # Seam masks to find a transition line between images with the least amount of interference
        seam_finder = SeamFinder()
        seam_masks = seam_finder.find(cropped_low_imgs, low_corners, cropped_low_masks)
        seam_masks = [
            seam_finder.resize(seam_mask, mask)
            for seam_mask, mask in zip(seam_masks, cropped_final_masks)
        ]

        # Exposure error compensation
        compensator = ExposureErrorCompensator()
        compensator.feed(low_corners, cropped_low_imgs, cropped_low_masks)
        compensated_imgs = [
            compensator.apply(idx, corner, img, mask)
            for idx, (img, mask, corner) in enumerate(
                zip(cropped_final_imgs, cropped_final_masks, final_corners)
            )
        ]

        # Blending
        blender = Blender()
        blender.prepare(final_corners, final_sizes)
        for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
            blender.feed(img, mask, corner)
        panorama, _ = blender.blend()

        return panorama
