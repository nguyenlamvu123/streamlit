import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform

from skimage.feature import ORB, match_descriptors, plot_matches
from skimage.transform import ProjectiveTransform
from skimage.measure import ransac
from skimage.feature import plot_matches


class Panorama:
    def __init__(
            self,
            in_dir: str = rf'M:{os.sep}streamlit{os.sep}pan',
            out_dir: str = rf'M:{os.sep}streamlit',
            debug: bool = True,
    ):
        self.debug = debug
        self.keypoints1 = None
        self.keypoints2 = None
        self.matches12 = None
        self.model_robust = None
        self.r = None
        self.c = None

        self.extension: tuple = (
            'jpg',
            # 'png',
            # 'JPG',
            # 'PNG'
        )

        self.out = out_dir
        self.load(in_dir)

    def load(self, in_dir):
        from skimage.color import rgb2gray

        p = [rf'{in_dir}{os.sep}*.{ex}' for ex in self.extension]
        ic = io.ImageCollection(p)

        if self.debug:
            self.compare(ic[0], ic[1], figsize=(10, 10))
        image0 = rgb2gray(ic[0][:, 500:500+1987, :])
        image1 = rgb2gray(ic[1][:, 500:500+1987, :])

        image0 = transform.rescale(image0, 0.25)
        image1 = transform.rescale(image1, 0.25)

        self.r, self.c = image1.shape[:2]

        if self.debug:
            self.compare(image0, image1)
        self.feature_detection_and_matching(image0, image1)
        self.transform_estimation(image0, image1)
        self.warping(image0, image1)
        print()

    @staticmethod
    def add_alpha(image, background=-1):
        from skimage.color import gray2rgb
        """Add an alpha layer to the image.
        The alpha layer is set to 1 for foreground and 0 for background.
        """
        return np.dstack((gray2rgb(image), (image != background)))

    @staticmethod
    def compare(*images, **kwargs):
        """
        Utility function to display images side by side.

        Parameters
        ----------
        image0, image1, image2, ... : ndarrray
            Images to display.
        labels : list
            Labels for the different images.
        """
        f, axes = plt.subplots(1, len(images), **kwargs)
        axes = np.array(axes, ndmin=1)

        labels = kwargs.pop('labels', None)
        if labels is None:
            labels = [''] * len(images)

        for n, (image, label) in enumerate(zip(images, labels)):
            axes[n].imshow(image, interpolation='nearest', cmap='gray')
            axes[n].set_title(label)
            axes[n].axis('off')

    def feature_detection_and_matching(self, image0, image1):  # TODO
        orb = ORB(n_keypoints=4000, fast_threshold=0.05)

        orb.detect_and_extract(image0)
        self.keypoints1 = orb.keypoints
        descriptors1 = orb.descriptors

        orb.detect_and_extract(image1)
        self.keypoints2 = orb.keypoints
        descriptors2 = orb.descriptors

        self.matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
        if self.debug:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            plot_matches(
                ax, image0, image1, self.keypoints1, self.keypoints2, self.matches12
            )
            ax.axis('off')

    def transform_estimation(self, image0, image1):  # TODO
        # Select keypoints from the source (image to be registered)
        # and target (reference image)
        src = self.keypoints2[self.matches12[:, 1]][:, ::-1]
        dst = self.keypoints1[self.matches12[:, 0]][:, ::-1]

        self.model_robust, inliers = ransac(
            (src, dst), ProjectiveTransform,
            min_samples=4, residual_threshold=1, max_trials=300
        )
        if self.debug:
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            plot_matches(ax, image0, image1, self.keypoints1, self.keypoints2, self.matches12[inliers])
            ax.axis('off')
            # plt.show()

    def warping(self, image0, image1):
        from skimage.transform import SimilarityTransform, warp
        from skimage.exposure import rescale_intensity

        # Note that transformations take coordinates in (x, y) format,
        # not (row, column), in order to be consistent with most literature
        corners = np.array([[0, 0],
                            [0, self.r],
                            [self.c, 0],
                            [self.c, self.r]])

        # Warp the image corners to their new positions
        warped_corners = self.model_robust(corners)

        # Find the extents of both the reference image and the warped
        # target image
        all_corners = np.vstack((warped_corners, corners))

        corner_min = np.min(all_corners, axis=0)
        corner_max = np.max(all_corners, axis=0)

        output_shape = (corner_max - corner_min)
        output_shape = np.ceil(output_shape[::-1])

        offset = SimilarityTransform(translation=-corner_min)

        image0_ = warp(image0, offset.inverse,
                       output_shape=output_shape, cval=-1)
        image1_ = warp(image1, (self.model_robust + offset).inverse,
                       output_shape=output_shape, cval=-1)

        image0_alpha = self.add_alpha(image0_)
        image1_alpha = self.add_alpha(image1_)

        merged = (image0_alpha + image1_alpha)
        alpha = merged[..., 3]

        # The summed alpha layers give us an indication of how many
        # images were combined to make up each pixel.  Divide by the
        # number of images to get an average.
        merged /= np.maximum(alpha, 1)[..., np.newaxis]
        if self.debug:
            self.compare(image0_alpha, image1_alpha, merged, figsize=(10, 10))


P = Panorama()