from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import remove_small_objects
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import minmax_scale


@dataclass
class RetinalSegmenter:
    bg_blur_ks: int = 5
    bg_blur_sigma: float = 10.0
    bg_canny_min_scale: float = 1
    bg_canny_max_scale: float = 2
    bg_canny_dilate: int = 5
    mask_erode: int = 20
    vessel_blur_ks: int = 5
    vessel_blur_sigma: float = 1.3
    vessel_dilate_ks: int = 3
    remove_small_obj_min_area: int = 64
    clahe_clip_limit: int = 2
    clahe_ks: int = 9

    def __post_init__(self):
        """ Correct datatypes after initialization """
        self.bg_blur_ks = int(self.bg_blur_ks)
        self.bg_canny_dilate = int(self.bg_canny_dilate)
        self.mask_erode = int(self.mask_erode)
        self.vessel_blur_ks = int(self.vessel_blur_ks)
        self.vessel_dilate_ks = int(self.vessel_dilate_ks)
        self.remove_small_obj_min_area = int(self.remove_small_obj_min_area)
        self.clahe_clip_limit = int(self.clahe_clip_limit)
        self.clahe_ks = int(self.clahe_ks)

    def predict(self, x: np.ndarray) -> np.ndarray:
        mask = self.segment_background(x)
        return self.segment_vessels(x, mask)

    def score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """ Returns the F1 Score between the pred and true image """
        return f1_score(y_pred.flatten(), y_true.flatten())

    def segment_background(self, x: np.ndarray) -> np.ndarray:
        """ Segment the background from the image through canny edge detection.

        Args:
            x: Input Retina Image

        Returns:
            A background mask with background as True
        """

        # Perform PCA to reduce dimensions to 1
        pca = PCA(1)
        shape = x.shape
        x_pca = pca.fit_transform(x.reshape(-1, 3))
        x_mms = minmax_scale(x_pca, (0, 255)).reshape(shape[:-1])

        # Reduce high intensity noise
        x_blur = cv2.GaussianBlur(x_mms, (self.bg_blur_ks, self.bg_blur_ks), self.bg_blur_sigma)

        # Perform Canny Edge Detection
        # We'll automatically set the canny thresholds via otsu's method
        canny_lower, canny_upper = (o := threshold_otsu(x_blur)) * 0.5, o

        x_canny = cv2.Canny(x_blur.astype(np.uint8), canny_lower, canny_upper)

        # Canny may not create a perfect loop
        x_canny_dil = cv2.dilate(
            x_canny,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (self.bg_canny_dilate, self.bg_canny_dilate))
        )

        # We'll extract the contours from the canny
        contours, _ = cv2.findContours(x_canny_dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

        # Find the biggest contour
        contour_max = max(contours, key=cv2.contourArea)

        mask = cv2.drawContours(
            np.zeros_like(x[..., 0]),
            [contour_max], -1, 1,
            thickness=cv2.FILLED
        ).astype(bool)
        return mask

    def segment_vessels(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """ Segment the vessels out of the retina.

        Args:
            x: Input Retina Image
            mask: Background mask

        Returns:
            Segmented Vessels
        """

        # While ridge filters can handle varying neighbourhood contrasts, still isn't perfect.
        # I found that the light spot on the Optic Nerve is susceptible to being mistaken by a vessel
        # We can improve the contrast of the image via CLAHE, which only works on 1 dimension.
        # Instead of using PCA, we use CLAHE on the brightness dimension after we convert RGB to LAB
        lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
        lab_planes = list(cv2.split(lab))
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(self.clahe_ks, self.clahe_ks))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        x_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # We blur again as ridge filters are susceptible to edges due to noise
        x_blur = cv2.GaussianBlur(x_clahe, (self.vessel_blur_ks, self.vessel_blur_ks), self.vessel_blur_sigma)

        # Apply the ridge filter
        ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create()
        x_ridge = ridge_filter.getRidgeFilteredImage(x_blur)

        # Erode to ensure that all edge ridges due to foreground/background are removed
        mask_eroded = cv2.erode(mask.astype(np.uint8), np.ones((self.mask_erode, self.mask_erode)))
        x_masked = x_ridge * mask_eroded

        # Yield high derivatives automatically using Otsu
        x_thresh = (x_masked > threshold_otsu(x_masked)).astype(np.uint8)

        # There are some broken vessel paths, we attempt to reconnect close ones by dilating with a circle kernel
        x_dilate = cv2.morphologyEx(
            x_thresh.astype(np.uint8), cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.vessel_dilate_ks, self.vessel_dilate_ks))
        )

        # Those paths that still fail to connect, we'll assume they are not a vessel unless huge.
        x_rso = remove_small_objects(x_dilate.astype(bool), self.remove_small_obj_min_area)

        return x_rso
