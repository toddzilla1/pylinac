import math
import typing
import random

import typing_extensions
import pathlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylinac


class LasVegasHelper(pylinac.planar_imaging.ImagePhantomBase):
    common_name = "Las Vegas Helper"
    phantom_bbox_size_mm2 = pylinac.LasVegas.phantom_bbox_size_mm2
    detection_conditions = pylinac.LasVegas.detection_conditions
    phantom_outline_object = pylinac.LasVegas.phantom_outline_object
    mtf = None

    # flip image left to right?
    flip_lr: bool = None

    # number of 90 degree rotations needed
    num_rot90: int = None

    # need to invert image?
    needs_inversion = False

    # Create some background regions of interest near our target locations
    low_contrast_background_roi_settings = {
        "roi 0": {"distance from center": 0.222, "angle": 0, "roi radius": 0.04},
        "roi 1": {"distance from center": 0.222, "angle": 90, "roi radius": 0.04},
        "roi 2": {"distance from center": 0.222, "angle": 180, "roi radius": 0.04},
        "roi 3": {"distance from center": 0.222, "angle": 270, "roi radius": 0.04},
    }

    # When properly oriented, the top-right ROI should have the highest contrast. Look for that region in 8 different
    # possible orientations.
    low_contrast_roi_settings = {
        "roi 0": {"distance from center": 0.179, "angle": -76.5, "roi radius": 0.016, "background": [3]},
        "roi 1": {"distance from center": 0.179, "angle": -103.5, "roi radius": 0.016, "background": [3]},
        "roi 2": {"distance from center": 0.179, "angle": -166.5, "roi radius": 0.016, "background": [2]},
        "roi 3": {"distance from center": 0.179, "angle": 166.5, "roi radius": 0.016, "background": [2]},
        "roi 4": {"distance from center": 0.179, "angle": 103.5, "roi radius": 0.016, "background": [1]},
        "roi 5": {"distance from center": 0.179, "angle": 76.5, "roi radius": 0.016, "background": [1]},
        "roi 6": {"distance from center": 0.179, "angle": 13.5, "roi radius": 0.016, "background": [0]},
        "roi 7": {"distance from center": 0.179, "angle": -13.5, "roi radius": 0.016, "background": [0]},
    }

    def _preprocess(self):
        # noinspection PyStatementEffect
        None

    def _phantom_radius_calc(self) -> float:
        return math.sqrt(self.phantom_ski_region.bbox_area) * 1.626

    def _phantom_angle_calc(self) -> float:
        return 0.0

    # override low contrast calculation to use local background ROIs if specified
    # noinspection PyUnresolvedReferences
    def _sample_low_contrast_rois(self) -> list[pylinac.core.roi.LowContrastDiskROI]:
        """Sample the low-contrast sample regions for calculating contrast values."""
        lc_rois = []
        for stng in self.low_contrast_roi_settings.values():
            roi = pylinac.core.roi.LowContrastDiskROI.from_phantom_center(
                self.image,
                self.phantom_angle + stng["angle"],
                self.phantom_radius * stng["roi radius"],
                self.phantom_radius * stng["distance from center"],
                self.phantom_center,
                self._low_contrast_threshold,
                self.low_contrast_background_value if not ("background" in stng)
                else (np.median([self.low_contrast_background_rois[roi].pixel_value for roi in stng["background"]])),
                contrast_method=self._low_contrast_method,
                visibility_threshold=self.visibility_threshold,
            )
            lc_rois.append(roi)
        return lc_rois

    # noinspection PyUnresolvedReferences
    def analyze(
            self,
            low_contrast_threshold: float = 0.05,
            high_contrast_threshold: float = 0.5,
            invert: bool = False,
            angle_override: float | None = None,
            center_override: tuple | None = None,
            size_override: float | None = None,
            ssd: float | typing.Literal["auto"] = "auto",
            low_contrast_method: str = pylinac.core.contrast.Contrast.MICHELSON,
            visibility_threshold: float = 100,
            x_adjustment: float = 0,
            y_adjustment: float = 0,
            angle_adjustment: float = 0,
            roi_size_factor: float = 1,
            scaling_factor: float = 1,) -> None:

        super().analyze()

        # get a list of the CNRs
        roi_data_dicts = self.results_data(as_dict=True)['low_contrast_rois']
        cnrs = [item['cnr'] for item in roi_data_dicts]

        # which orientation has the max CNR?
        max_roi = cnrs.index(max(cnrs))

        # need to invert image?
        self.needs_inversion = self.low_contrast_background_value < self.low_contrast_rois[max_roi].pixel_value

        # define flip and rotation depending on max roi
        match max_roi:
            case 0:
                self.flip_lr = False
                self.num_rot90 = 0

            case 1:
                self.flip_lr = True
                self.num_rot90 = 0

            case 2:
                self.flip_lr = False
                self.num_rot90 = 3

            case 3:
                self.flip_lr = True
                self.num_rot90 = 1

            case 4:
                self.flip_lr = False
                self.num_rot90 = 2

            case 5:
                self.flip_lr = True
                self.num_rot90 = 2

            case 6:
                self.flip_lr = False
                self.num_rot90 = 1

            case 7:
                self.flip_lr = True
                self.num_rot90 = 3


class LasVegasUpdated(pylinac.LasVegas):
    local_low_contrast_background_roi_settings = {
        "roi 0": {"distance from center": 0.24, "angle": 241, "roi radius": 0.028},
        "roi 1": {"distance from center": 0.218, "angle": 256, "roi radius": 0.028},
        "roi 2": {"distance from center": 0.212, "angle": 272, "roi radius": 0.028},
        "roi 3": {"distance from center": 0.174, "angle": 228, "roi radius": 0.028},
        "roi 4": {"distance from center": 0.14, "angle": 248, "roi radius": 0.028},
        "roi 5": {"distance from center": 0.13, "angle": 274, "roi radius": 0.028},
        "roi 6": {"distance from center": 0.068, "angle": 217, "roi radius": 0.028},
        "roi 7": {"distance from center": 0.04, "angle": 274, "roi radius": 0.028},
        "roi 8": {"distance from center": 0.075, "angle": 325, "roi radius": 0.02},
        "roi 9": {"distance from center": 0.068, "angle": 143, "roi radius": 0.028},
        "roi 10": {"distance from center": 0.04, "angle": 86, "roi radius": 0.028},
        "roi 11": {"distance from center": 0.075, "angle": 35, "roi radius": 0.02},
        "roi 12": {"distance from center": 0.14, "angle": 112, "roi radius": 0.028},
        "roi 13": {"distance from center": 0.13, "angle": 86, "roi radius": 0.028},
        "roi 14": {"distance from center": 0.148, "angle": 63, "roi radius": 0.02},
        "roi 15": {"distance from center": 0.22, "angle": 104, "roi radius": 0.028},
        "roi 16": {"distance from center": 0.21, "angle": 88, "roi radius": 0.028},
        "roi 17": {"distance from center": 0.222, "angle": 73, "roi radius": 0.02},
        "roi 18": {"distance from center": 0.18, "angle": 0, "roi radius": 0.028},
        "roi 19": {"distance from center": 0.20, "angle": 26, "roi radius": 0.028},
        "roi 20": {"distance from center": 0.25, "angle": 44, "roi radius": 0.028},
    }

    local_low_contrast_roi_settings = {
        "roi 1": {"distance from center": 0.107, "angle": 0.5, "roi radius": 0.028, "background": [8, 11, 18]},
        "roi 2": {"distance from center": 0.141, "angle": 39.5, "roi radius": 0.028, "background": [11, 14, 19]},
        "roi 3": {"distance from center": 0.205, "angle": 58, "roi radius": 0.028, "background": [14, 17, 20]},
        "roi 4": {"distance from center": 0.179, "angle": -76.5, "roi radius": 0.016, "background": [2, 5]},
        "roi 5": {"distance from center": 0.095, "angle": -63.5, "roi radius": 0.016, "background": [5, 8]},
        "roi 6": {"distance from center": 0.042, "angle": 0.5, "roi radius": 0.016, "background": [7, 8, 10, 11]},
        "roi 7": {"distance from center": 0.097, "angle": 65.5, "roi radius": 0.016, "background": [10, 11, 13, 14]},
        "roi 8": {"distance from center": 0.178, "angle": 76.5, "roi radius": 0.016, "background": [13, 14, 16, 17]},
        "roi 9": {"distance from center": 0.174, "angle": -97.5, "roi radius": 0.012, "background": [1, 2, 4, 5]},
        "roi 10": {"distance from center": 0.088, "angle": -105.5, "roi radius": 0.012, "background": [4, 5, 6, 7]},
        "roi 11": {"distance from center": 0.024, "angle": -183.5, "roi radius": 0.012, "background": [6, 7, 9, 10]},
        "roi 12": {"distance from center": 0.091, "angle": 105.5, "roi radius": 0.012, "background": [9, 10, 12, 13]},
        "roi 13": {"distance from center": 0.179, "angle": 97.5, "roi radius": 0.012, "background": [12, 13, 15, 16]},
        "roi 14": {"distance from center": 0.189, "angle": -113.5, "roi radius": 0.007, "background": [1, 4]},
        "roi 15": {"distance from center": 0.113, "angle": -131.5, "roi radius": 0.007, "background": [4, 6]},
        "roi 16": {"distance from center": 0.0745, "angle": -181.5, "roi radius": 0.007, "background": [6, 9]},
        "roi 17": {"distance from center": 0.115, "angle": 130, "roi radius": 0.007, "background": [9, 12]},
        "roi 18": {"distance from center": 0.191, "angle": 113, "roi radius": 0.007, "background": [12, 15]},
        "roi 19": {"distance from center": 0.2085, "angle": -124.6, "roi radius": 0.003, "background": [0, 3]},
        "roi 20": {"distance from center": 0.146, "angle": -144.3, "roi radius": 0.003, "background": [3]}
    }

    # turn off the built-in inversion check
    def _check_inversion(self):
        pass

    def _calc_background_weighted_average(self, roi_indexes: typing.List[int]) -> float:
        """Average pixel value of the specified background ROIs, weighted by area"""
        values = [self.low_contrast_background_rois[roi_index].pixel_value for roi_index in roi_indexes]
        weights = [self.low_contrast_background_rois[roi_index].area for roi_index in roi_indexes]
        return np.average(values, weights=weights)

    # override low contrast calculation to use local background ROIs if specified
    # noinspection PyUnresolvedReferences
    def _sample_low_contrast_rois(self) -> list[pylinac.core.roi.LowContrastDiskROI]:
        """Sample the low-contrast sample regions for calculating contrast values."""
        lc_rois = []
        for stng in self.low_contrast_roi_settings.values():
            # noinspection DuplicatedCode
            roi = pylinac.core.roi.LowContrastDiskROI.from_phantom_center(
                self.image,
                self.phantom_angle + stng["angle"],
                self.phantom_radius * stng["roi radius"],
                self.phantom_radius * stng["distance from center"],
                self.phantom_center,
                self._low_contrast_threshold,
                self.low_contrast_background_value if not ("background" in stng)
                else self._calc_background_weighted_average(stng["background"]),
                contrast_method=self._low_contrast_method,
                visibility_threshold=self.visibility_threshold,
            )
            lc_rois.append(roi)
        return lc_rois

    # noinspection PyUnresolvedReferences
    @classmethod
    def create_analyzed_phantom(
            cls: typing_extensions.Self,
            filepath: str | typing.BinaryIO | pathlib.Path,
            fix_orientation: bool = True,
            auto_invert: bool = True,
            use_local_background: bool = True,
            low_contrast_method: str = pylinac.core.contrast.Contrast.MICHELSON,
    ) -> typing_extensions.Self:
        """ Factory method for fetching analyzed phantoms with results in place.

        Parameters
        ----------
        filepath: str
            Path to the image file.
        fix_orientation: bool
            Whether to attempt to rotate and/or flip the image into the expected orientation.
        auto_invert: bool
            Whether to automatically invert the image if needed
        use_local_background: bool
            Whether to use custom background ROIs near each target ROI or the default four spaced around the perimeter
        low_contrast_method: str
            Which contrast algorithm to  use
        """
        # create the phantom
        phantom = LasVegasUpdated(filepath)

        # create helper class
        lv_helper = LasVegasHelper(filepath)
        lv_helper.analyze()

        # fix orientation if requested
        if fix_orientation:
            if lv_helper.flip_lr:
                phantom.image.fliplr()
            if lv_helper.num_rot90 > 0:
                phantom.image.rot90(lv_helper.num_rot90)

        # use local background if requested
        if use_local_background:
            phantom.low_contrast_roi_settings = cls.local_low_contrast_roi_settings
            phantom.low_contrast_background_roi_settings = cls.local_low_contrast_background_roi_settings

        # process phantom
        phantom.analyze(
            invert=lv_helper.needs_inversion if auto_invert else False,
            low_contrast_method=low_contrast_method,
        )

        return phantom

    @property
    def total_visibility(self) -> float:
        """ Sum of all ROI visibilities. Possibly integrate with superclass generate_results() """

        return sum(roi.visibility for roi in self.low_contrast_rois)

    def _plot_image_basic(self, axes: plt.Axes):
        """Basic, grayscale image of the phantom x-ray"""
        self.image.plot(
            ax=axes,
            show=False,
            vmin=self.window_floor(),
            vmax=self.window_ceiling(),
        )
        axes.axis("off")
        axes.set_title(f"Source Image")

    def _plot_image_details(self, axes: plt.Axes):
        """Image enhanced to show details"""
        axes.imshow(
            self.image.array,
            vmin=self.window_floor(),
            vmax=self.window_ceiling(),
            cmap="flag",
        )
        axes.axis("off")
        axes.set_title(f"Source Image, Details")

    def _plot_image_with_visible_ROIs(self, axes: plt.Axes, background_rois: bool):
        """Image annotated with ROIs"""
        self.image.plot(
            ax=axes,
            show=False,
            vmin=self.window_floor(),
            vmax=self.window_ceiling(),
        )
        axes.axis("off")
        axes.set_title(f"ROI Visibility (Pass / Fail)")

        # plot the outline image
        if self.phantom_outline_object is not None:
            outline_obj, settings = self._create_phantom_outline_object()
            outline_obj.plot2axes(axes, edgecolor="b")
        # plot the low contrast background ROIs
        if background_rois:
            for roi in self.low_contrast_background_rois:
                roi.plot2axes(axes, edgecolor="b")
        # plot the low contrast ROIs
        for roi in self.low_contrast_rois:
            roi.plot2axes(axes, edgecolor=roi.plot_color)

    def _plot_image_with_visible_ROIs_scaled(self, axes: plt.Axes):
        """Image annotated with low contrast ROIs, colorscaled"""
        self.image.plot(
            ax=axes,
            show=False,
            vmin=self.window_floor(),
            vmax=self.window_ceiling(),
        )
        axes.axis("off")
        axes.set_title(f"ROI Visibility (Continuous)")

        # plot the outline image
        if self.phantom_outline_object is not None:
            outline_obj, settings = self._create_phantom_outline_object()
            outline_obj.plot2axes(axes, edgecolor="b")

        # find max visibility
        max_visibility = 0.0
        for roi in self.low_contrast_rois:
            if roi.visibility > max_visibility:
                max_visibility = roi.visibility

        # define visibility colormap
        cmap = mpl.colormaps['RdYlGn']
        norm = mpl.colors.TwoSlopeNorm(vmin=0, vmax=max_visibility, vcenter=self.visibility_threshold)

        # plot the low contrast ROIs
        for roi in self.low_contrast_rois:
            roi.plot2axes(axes, edgecolor=cmap(norm(roi.visibility)))

        # show colorbar
        axes.get_figure().colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes, label=f"Visibility (threshold = {self.visibility_threshold})")

    def plot_las_vegas_custom_results(
            self,
            show: bool = True,
    ) -> tuple[list[plt.Figure], list[str]]:
        """Plot custom results for the las vegas analysis.

        Parameters
        ----------
        show : bool
            Whether to show the image when called.
        """
        num_plots = 6

        # set up axes and make axes iterable
        figs = []
        names = []
        axes = []
        for n in range(num_plots):
            fig, axis = plt.subplots(1)
            figs.append(fig)
            axes.append(axis)

        if num_plots < 2:
            axes = (axes,)
        axes = iter(axes)

        # plot the basic image
        names.append("image")
        self._plot_image_basic(next(axes))

        # plot image with enhanced details
        names.append("image details")
        self._plot_image_details(next(axes))

        # plot image with background ROIs
        names.append("ROIs with Background")
        self._plot_image_with_visible_ROIs(next(axes), background_rois=True)

        # plot image without background ROIs
        names.append("ROIs")
        self._plot_image_with_visible_ROIs(next(axes), background_rois=False)

        # plot image with background ROIs using colorscale
        names.append("ROIs colorscale")
        self._plot_image_with_visible_ROIs_scaled(next(axes))

        # plot the low contrast value graph
        names.append("low_contrast")
        self._plot_lowcontrast_graph(next(axes))

        plt.tight_layout()
        if show:
            plt.show()
        return figs, names

    def plot_background_roi_connections(self):
        """Show which background ROI is connected to each low contrast ROI"""
        fig, axes = plt.subplots(1)

        # plot image
        self.image.plot(
            ax=axes,
            show=False,
            vmin=self.window_floor(),
            vmax=self.window_ceiling(),
        )
        axes.axis("off")
        axes.set_title("Background ROI Connections")

        # plot the outline image
        if self.phantom_outline_object is not None:
            outline_obj, settings = self._create_phantom_outline_object()
            outline_obj.plot2axes(axes, edgecolor="b")

        # plot the low contrast background ROIs
        for roi in self.low_contrast_background_rois:
            roi.plot2axes(axes, edgecolor="b")

        # plot the low contrast ROIs
        for roi in self.low_contrast_rois:
            roi.plot2axes(axes, edgecolor="g")

        # create a colormap for the connection lines
        cmap = mpl.colormaps['rainbow']
        colors = [cmap(float(i) / len(self.low_contrast_roi_settings)) for i in
                  range(0, len(self.low_contrast_roi_settings))]
        random.shuffle(colors)
        colors = iter(colors)

        # plot the connections
        for stng in self.low_contrast_roi_settings.values():
            if "background" in stng:
                # get connected ROIs
                roi_indexes = stng["background"]

                # create target ROI
                roi = pylinac.core.roi.LowContrastDiskROI.from_phantom_center(
                    self.image,
                    self.phantom_angle + stng["angle"],
                    self.phantom_radius * stng["roi radius"],
                    self.phantom_radius * stng["distance from center"],
                    self.phantom_center,
                    self._low_contrast_threshold,
                    self.low_contrast_background_value if not ("background" in stng)
                    else self._calc_background_weighted_average(stng["background"]),
                    contrast_method=self._low_contrast_method,
                    visibility_threshold=self.visibility_threshold,
                )

                # plot connections
                clr = next(colors)
                for roi_index in roi_indexes:
                    roi_background = self.low_contrast_background_rois[roi_index]
                    axes.plot([roi.center.x, roi_background.center.x], [roi.center.y, roi_background.center.y],
                              color=clr, linewidth=0.8)

        plt.show()