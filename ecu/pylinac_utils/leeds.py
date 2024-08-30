import dataclasses
import functools
import math
import random
import typing
import warnings

import plotly.express as px
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import py_linq
import skimage

import pylinac
import pylinac.core.roi

import ecu.pylinac_utils.roi


@dataclasses.dataclass(frozen=True)
class HighContrastAlignedRectangularROI():
    image_array: np.ndarray
    contrast_threshold: float
    # noinspection PyUnresolvedReferences
    center_point_px: pylinac.core.geometry.Point
    width_px: float
    height_px: float
    angular_direction_rad: float
    nominal_line_pairs_per_mm: float
    nominal_amplitude: float

    @functools.cached_property
    def _delta_x1(self) -> float:
        return (self.height_px / 2.0) * math.cos(self.angular_direction_rad)

    @functools.cached_property
    def _delta_y1(self) -> float:
        return (self.height_px / 2.0) * math.sin(self.angular_direction_rad)

    @functools.cached_property
    def _delta_x2(self) -> float:
        return (self.width_px / 2.0) * math.sin(self.angular_direction_rad)

    @functools.cached_property
    def _delta_y2(self) -> float:
        return (self.width_px / 2.0) * math.cos(self.angular_direction_rad)

    # noinspection PyUnresolvedReferences
    @functools.cached_property
    def pt_btm_middle_px(self) -> pylinac.core.geometry.Point:
        return pylinac.core.geometry.Point(
            self.center_point_px.x - self._delta_x1,
            self.center_point_px.y - self._delta_y1
        )

    # noinspection PyUnresolvedReferences
    @functools.cached_property
    def pt_btm_left_px(self) -> pylinac.core.geometry.Point:
        return pylinac.core.geometry.Point(
            self.pt_btm_middle_px.x + self._delta_x2,
            self.pt_btm_middle_px.y - self._delta_y2
        )

    # noinspection PyUnresolvedReferences
    @functools.cached_property
    def pt_btm_right_px(self) -> pylinac.core.geometry.Point:
        return pylinac.core.geometry.Point(
            self.pt_btm_middle_px.x - self._delta_x2,
            self.pt_btm_middle_px.y + self._delta_y2
        )

    # noinspection PyUnresolvedReferences
    @functools.cached_property
    def pt_top_middle_px(self) -> pylinac.core.geometry.Point:
        return pylinac.core.geometry.Point(
            self.center_point_px.x + self._delta_x1,
            self.center_point_px.y + self._delta_y1
        )

    # noinspection PyUnresolvedReferences
    @functools.cached_property
    def pt_top_left_px(self) -> pylinac.core.geometry.Point:
        return pylinac.core.geometry.Point(
            self.pt_top_middle_px.x + self._delta_x2,
            self.pt_top_middle_px.y - self._delta_y2
        )

    # noinspection PyUnresolvedReferences
    @functools.cached_property
    def pt_top_right_px(self) -> pylinac.core.geometry.Point:
        return pylinac.core.geometry.Point(
            self.pt_top_middle_px.x - self._delta_x2,
            self.pt_top_middle_px.y + self._delta_y2
        )

    @functools.cached_property
    def masked_image(self) -> np.ndarray:
        rr, cc = skimage.draw.polygon(
            [self.pt_btm_left_px.y, self.pt_top_left_px.y, self.pt_top_right_px.y, self.pt_btm_right_px.y],
            [self.pt_btm_left_px.x, self.pt_top_left_px.x, self.pt_top_right_px.x, self.pt_btm_right_px.x]
        )
        return self.image_array[rr, cc]

    @functools.cached_property
    def mean(self) -> float:
        return np.mean(self.masked_image)

    @functools.cached_property
    def std(self) -> float:
        return np.std(self.masked_image)

    @functools.cached_property
    def profile(self) -> np.ndarray:
        """Intensity profile of the ROI along the angular_direction.
        """
        return skimage.measure.profile_line(
            image=self.image_array,
            src=[self.pt_btm_middle_px.y, self.pt_btm_middle_px.x],
            dst=[self.pt_top_middle_px.y, self.pt_top_middle_px.x],
            linewidth=int(self.width_px),
        )

    @functools.cached_property
    def mtf(self) -> float:
        """Modulation Transfer Function for the ROI

        González-López, Campos-Morcillo, and Lago-Martín: MTF calculation from a bar pattern phantom
        http://dx.doi.org/10.1118/1.4963211

        Page 5656, Equation 8
        """

        # Start with Equation 9 ('j' denotes the imaginary part of a complex number in Python)
        N = len(self.profile)
        f1n = sum(self.profile[k - 1] * np.exp(-2.0j * np.pi * k / N) for k in range(1, N + 1)) / N

        # Now Equation 8. Note: the absolute value of a complex number is its vector length in the complex plane
        return np.pi * np.abs(f1n) / (2.0 * self.nominal_amplitude)

    def plot2axes(
            self,
            axes: plt.Axes | None = None,
            edgecolor: str = "black",
            fill: bool = False,
            text: str = "",
            fontsize: str = "medium",
            **kwargs,
    ) -> None:
        """Plot the ROI on the axes.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            An MPL axes to plot to.
        edgecolor : str
            The color of the circle.
        fill : bool
            Whether to fill the circle with color or leave hollow.
        text: str
            If provided, plots the given text at the center. Useful for differentiating ROIs on a plotted image.
        fontsize: str
            The size of the text, if provided. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html
            for options.
        """
        if axes is None:
            fig, axes = plt.subplots()
            axes.imshow(self.image_array)

        axes.add_patch(
            matplotlib.patches.Polygon(
                ((self.pt_btm_left_px.x, self.pt_btm_left_px.y),
                 (self.pt_top_left_px.x, self.pt_top_left_px.y),
                 (self.pt_top_right_px.x, self.pt_top_right_px.y),
                 (self.pt_btm_right_px.x, self.pt_btm_right_px.y)),
                closed=True,
                edgecolor=edgecolor,
                fill=fill,
                **kwargs,
            )
        )

        if text:
            axes.text(
                x=self.center_point_px.x,
                y=self.center_point_px.y,
                s=text,
                fontsize=fontsize,
                color=edgecolor,
            )


class SampledProfileMTF(pylinac.core.mtf.MTF):
    """Calculates the modulation transfer function (MTF) of a sampled profile."""

    def __init__(self, rois: typing.List[HighContrastAlignedRectangularROI]):
        self.rois = rois

        # Format some data as expected by the superclass
        self.spacings = [roi.nominal_line_pairs_per_mm for roi in self.rois]
        if len(self.spacings) < 2:
            raise ValueError(
                "The number of MTF spacings must be greater than 1."
            )

        self.mtfs = {}
        for roi in self.rois:
            self.mtfs[roi.nominal_line_pairs_per_mm] = roi.mtf

        # sort according to spacings
        self.mtfs = {k: v for k, v in sorted(self.mtfs.items(), key=lambda x: x[0])}

        # normalize to first region
        self.norm_mtfs = {}
        for key, value in self.mtfs.items():
            self.norm_mtfs[key] = (
                    value / list(self.mtfs.values())[0]
            )

        # check that the MTF drops monotonically by measuring the deltas between MTFs
        # if the delta is increasing it means the MTF rose on a subsequent value
        max_delta = np.max(np.diff(list(self.norm_mtfs.values())))
        if max_delta > 0:
            warnings.warn(
                "The MTF does not drop monotonically; be sure the ROIs are correctly aligned."
            )


class LeedsTORUpdated(pylinac.LeedsTOR):
    _roi_r1 = 0.77
    _roi_r2 = 0.03

    # low_contrast_background_roi inner and outer radii
    low_contrast_background_roi_inner_radius = 0.07
    low_contrast_background_roi_outer_radius = 0.13

    # This class implements a custom low_contrast_background_roi (annulus) centered on each low_contrast_roi
    low_contrast_background_roi_settings = None

    # Prepend "custom_" to trigger custom roi sampling
    low_contrast_roi_settings = None
    custom_low_contrast_roi_settings = {
        # set 1
        "roi 0": {"distance from center": _roi_r1, "angle": 30, "roi radius": _roi_r2},
        "roi 1": {"distance from center": _roi_r1, "angle": 45, "roi radius": _roi_r2},
        "roi 2": {"distance from center": _roi_r1, "angle": 60, "roi radius": _roi_r2},
        "roi 3": {"distance from center": _roi_r1, "angle": 75, "roi radius": _roi_r2},
        "roi 4": {"distance from center": _roi_r1, "angle": 90, "roi radius": _roi_r2},
        "roi 5": {"distance from center": _roi_r1, "angle": 105, "roi radius": _roi_r2},
        "roi 6": {"distance from center": _roi_r1, "angle": 120, "roi radius": _roi_r2},
        "roi 7": {"distance from center": _roi_r1, "angle": 135, "roi radius": _roi_r2},
        "roi 8": {"distance from center": _roi_r1, "angle": 150, "roi radius": _roi_r2},
        # set 2
        "roi 9": {"distance from center": _roi_r1, "angle": 210, "roi radius": _roi_r2},
        "roi 10": {"distance from center": _roi_r1, "angle": 225, "roi radius": _roi_r2},
        "roi 11": {"distance from center": _roi_r1, "angle": 240, "roi radius": _roi_r2},
        "roi 12": {"distance from center": _roi_r1, "angle": 255, "roi radius": _roi_r2},
        "roi 13": {"distance from center": _roi_r1, "angle": 270, "roi radius": _roi_r2},
        "roi 14": {"distance from center": _roi_r1, "angle": 285, "roi radius": _roi_r2},
        "roi 15": {"distance from center": _roi_r1, "angle": 300, "roi radius": _roi_r2},
        "roi 16": {"distance from center": _roi_r1, "angle": 315, "roi radius": _roi_r2},
        "roi 17": {"distance from center": _roi_r1, "angle": 330, "roi radius": _roi_r2},
    }

    # prepending "aligned_" here allows us to avoid the ImagePhantomBase.analyze() reference to normal high contrast
    # ROI processing and implement our own
    high_contrast_roi_settings = None
    aligned_high_contrast_roi_settings = {
        "roi 1": {
            "distance from center": 0.2895,
            "angle": 54.62,
            "width": 0.10,
            "height": 0.10,
            "direction": 0.0,
            "lp/mm": 0.5,
        },
        "roi 2": {
            "distance from center": 0.187,
            "angle": 25.1,
            "width": 0.1,
            "height": 0.093,
            "direction": 0.0,
            "lp/mm": 0.56,
        },
        "roi 3": {
            "distance from center": 0.1848,
            "angle": 335.5,
            "width": 0.1,
            "height": 0.084,
            "direction": 0.0,
            "lp/mm": 0.63,
        },
        "roi 4": {
            "distance from center": 0.238,
            "angle": 80.06,
            "width": 0.1,
            "height": 0.072,
            "direction": 0.0,
            "lp/mm": 0.71,
        },
        "roi 5": {
            "distance from center": 0.0916,
            "angle": 62.96,
            "width": 0.1,
            "height": 0.064,
            "direction": 0.0,
            "lp/mm": 0.8,
        },
        "roi 6": {
            "distance from center": 0.093,
            "angle": -64,
            "width": 0.1,
            "height": 0.064,
            "direction": 0.0,
            "lp/mm": 0.9,
        },
        "roi 7": {
            "distance from center": 0.239,
            "angle": 101.98,
            "width": 0.1,
            "height": 0.050,
            "direction": 0.0,
            "lp/mm": 1.0,
        },
        "roi 8": {
            "distance from center": 0.0907,
            "angle": 122.62,
            "width": 0.1,
            "height": 0.048,
            "direction": 0.0,
            "lp/mm": 1.12,
        },
        "roi 9": {
            "distance from center": 0.09515,
            "angle": 239.07,
            "width": 0.1,
            "height": 0.040,
            "direction": 0.0,
            "lp/mm": 1.25,
        },
        "roi 10": {
            "distance from center": 0.2596,
            "angle": 115.8,
            "width": 0.1,
            "height": 0.032,
            "direction": 0.0,
            "lp/mm": 1.4,
        },
        "roi 11": {
            "distance from center": 0.138,
            "angle": 145,
            "width": 0.1,
            "height": 0.028,
            "direction": 0.0,
            "lp/mm": 1.6,
        },
        "roi 12": {
            "distance from center": 0.13967,
            "angle": 216.4,
            "width": 0.1,
            "height": 0.022,
            "direction": 0.0,
            "lp/mm": 1.8,
        },
    }

    # Override to return rectangular ROIs instead of disc
    def _sample_high_contrast_rois(self) -> list[HighContrastAlignedRectangularROI]:
        """Sample the high-contrast line pair regions. We overload to find
        the center of the high-res block which can be offset relative
        to the center depending on the model"""
        # find the high-res block ROI
        regions = self._get_canny_regions()
        high_res_block_size = self.phantom_bbox_size_px * 0.23
        sorted_regions = (
            py_linq.Enumerable(regions)
            .where(
                lambda r: math.isclose(r.bbox_area, high_res_block_size, rel_tol=0.75)
            )
            .where(
                lambda r: pylinac.roi.bbox_center(r).distance_to(self.phantom_center)
                          < 0.1 * self.phantom_radius
            )
            .order_by_descending(
                lambda r: pylinac.roi.bbox_center(r).distance_to(self.phantom_center)
            )
            .to_list()
        )
        if not sorted_regions:
            raise ValueError(
                "Could not find high-resolution block within the leeds phantom. Try rotating the image."
            )
        high_res_center = pylinac.roi.bbox_center(sorted_regions[0])
        self.high_res_center = high_res_center

        # do the same as the base method but centered on the high-res block
        # OVERRIDDEN to use HighContrastAlignedRectangularROI
        hc_rois = []
        for stng in self.aligned_high_contrast_roi_settings.values():
            location_angle_deg = self.phantom_angle + stng["angle"]
            location_radius_px = self.phantom_radius * stng["distance from center"]
            # noinspection PyUnresolvedReferences
            hc_roi = HighContrastAlignedRectangularROI(
                image_array=self.image.array,
                center_point_px=pylinac.core.geometry.Point(
                    x=high_res_center.x + np.cos(np.deg2rad(location_angle_deg)) * location_radius_px,
                    y=high_res_center.y + np.sin(np.deg2rad(location_angle_deg)) * location_radius_px,
                ),
                width_px=stng["width"] * self.phantom_radius,
                height_px=stng["height"] * self.phantom_radius,
                angular_direction_rad=np.deg2rad(stng["direction"] + self.phantom_angle),
                contrast_threshold=self._high_contrast_threshold,
                nominal_line_pairs_per_mm=stng["lp/mm"],
                nominal_amplitude=1.0  # TODO
            )
            hc_rois.append(hc_roi)
        return hc_rois

    # override to implement custom high contrast analysis and custom low_contrast_background_roi (annulus)
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
    ) -> None:
        """Analyze the phantom using the provided thresholds and settings."""

        # call superclass.analyze(). high contrast settings ignored since we are using different name for settings
        super().analyze(
            low_contrast_threshold=low_contrast_threshold,
            high_contrast_threshold=high_contrast_threshold,
            invert=invert,
            angle_override=angle_override,
            center_override=center_override,
            size_override=size_override,
            ssd=ssd,
            low_contrast_method=low_contrast_method,
            visibility_threshold=visibility_threshold,
        )

        # Custom low contrast ROI processing
        self.low_contrast_background_rois, self.low_contrast_rois = self._sample_low_contrast_rois()

        # Custom high contrast ROIs
        self.high_contrast_rois = self._sample_high_contrast_rois()

        # generate rMTF
        self.mtf = SampledProfileMTF(
            rois=self.high_contrast_rois
        )

        # Override to use annulus ROIs

    # override low contrast calculation to use local background ROIs if specified
    def _sample_low_contrast_rois(self) -> tuple[
        list[ecu.pylinac_utils.roi.AnnulusROI], list[pylinac.core.roi.LowContrastDiskROI]]:
        """Sample the low-contrast sample regions for calculating contrast values."""
        bg_rois = []
        lc_rois = []
        for stng in self.custom_low_contrast_roi_settings.values():
            bg_roi = ecu.pylinac_utils.roi.AnnulusROI(
                image_array=self.image.array,
                angle_deg=self.phantom_angle + stng["angle"],
                dist_from_center_px=self.phantom_radius * stng["distance from center"],
                roi_inner_radius_px=self.phantom_radius * self.low_contrast_background_roi_inner_radius,
                roi_outer_radius_px=self.phantom_radius * self.low_contrast_background_roi_outer_radius,
                phantom_center_pt_px=self.phantom_center,
            )
            bg_rois.append(bg_roi)

            lc_roi = pylinac.core.roi.LowContrastDiskROI(
                array=self.image.array,
                angle=self.phantom_angle + stng["angle"],
                roi_radius=self.phantom_radius * stng["roi radius"],
                dist_from_center=self.phantom_radius * stng["distance from center"],
                phantom_center=self.phantom_center,
                contrast_threshold=self._low_contrast_threshold,
                contrast_reference=bg_roi.pixel_value,
                contrast_method=self._low_contrast_method,
                visibility_threshold=self.visibility_threshold,
            )
            lc_rois.append(lc_roi)
        return bg_rois, lc_rois

    @property
    def total_contrast(self) -> float:
        """ Sum of all ROI contrasts. Possibly integrate with superclass generate_results() """

        return sum(roi.contrast for roi in self.low_contrast_rois)

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

    def _plot_image_with_ROIs(self, axes: plt.Axes, background_rois: bool):
        """Image annotated with ROIs"""

        self.image.plot(
            ax=axes,
            show=False,
            vmin=self.window_floor(),
            vmax=self.window_ceiling(),

            # TODO: make bigger
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
        # plot the high contrast ROIs
        for roi in self.high_contrast_rois:
            roi.plot2axes(axes, edgecolor='yellow')

    def _plot_image_with_visible_ROIs_scaled(self, axes: plt.Axes):
        """Image annotated with low contrast ROIs, colorscaled"""
        self.image.plot(
            ax=axes,
            show=False,
            vmin=self.window_floor(),
            vmax=self.window_ceiling(),
        )
        axes.axis("off")
        axes.set_title(f"ROI Contrast (Continuous)")

        # plot the outline image
        if self.phantom_outline_object is not None:
            outline_obj, settings = self._create_phantom_outline_object()
            outline_obj.plot2axes(axes, edgecolor="b")

        # find max contrast
        max_contrast = np.max([roi.contrast for roi in self.low_contrast_rois])

        # define colormap
        cmap = mpl.colormaps['RdYlGn']
        norm = mpl.colors.TwoSlopeNorm(vmin=0, vmax=max_contrast, vcenter=self._low_contrast_threshold)

        # plot the low contrast ROIs
        for roi in self.low_contrast_rois:
            roi.plot2axes(axes, edgecolor=cmap(norm(roi.contrast)))

        # show colorbar
        axes.get_figure().colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes, label=f"Contrast (threshold = {self._low_contrast_threshold})")

    def plot_leeds_custom_results(
            self,
            show: bool = True,
    ) -> tuple[list[plt.Figure], list[str]]:
        """Plot custom results for the Leeds TOR analysis.

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
        figs[2].set_dpi(300)
        self._plot_image_with_ROIs(next(axes), background_rois=True)

        # plot image without background ROIs
        # names.append("ROIs")
        # self._plot_image_with_visible_ROIs(next(axes), background_rois=False)

        # plot image with background ROIs using colorscale
        names.append("ROIs colorscale")
        self._plot_image_with_visible_ROIs_scaled(next(axes))

        # plot the low contrast value graph
        names.append("low_contrast")
        self._plot_lowcontrast_graph(next(axes), )

        # plot the high contrast value graph
        names.append("high_contrast")
        self._plot_highcontrast_graph(next(axes))

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
                roi = pylinac.core.roi.LowContrastDiskROI(
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

    def plot_interactive_image_details(self, show: bool = True) -> go.Figure:
        """ Create interactive figure of the image with a colorscale showing image details. """

        # Create image
        fig = px.imshow(
            img=self.image.array,
            color_continuous_scale=px.colors.qualitative.Alphabet,
            height=self.image.array.shape[0],
            width=self.image.array.shape[1],
        )

        # No colorbar
        fig.update_layout(coloraxis_showscale=False)

        # Display figure if requested
        if show:
            fig.show()

        return fig


    def plot_interactive_rois(self, show: bool = True) -> go.Figure:
        """ Create interactive figure of the ROIs for Leeds TOR analysis. """

        # Create image
        fig = px.imshow(
            img=self.image.array,
            color_continuous_scale='gray',
            height=self.image.array.shape[0],
            width=self.image.array.shape[1],
        )

        # No colorbar
        fig.update_layout(coloraxis_showscale=False)

        # Phantom outline and center
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=self.phantom_center.x - self.phantom_radius,
            y0=self.phantom_center.y - self.phantom_radius,
            x1=self.phantom_center.x + self.phantom_radius,
            y1=self.phantom_center.y + self.phantom_radius,
            line_color="Blue",
            showlegend=True,
            legendgroup='Phantom Elements',
            legendgrouptitle={'text': 'Phantom Elements'},
            name='Outline',
            legendrank=1,
        )

        #
        # Add low contrast ROIs
        #

        # find max contrast
        max_contrast = np.max([roi.contrast for roi in self.low_contrast_rois])

        # define colormap (borrow from matplotlib)
        cmap = mpl.colormaps['RdYlGn']
        norm = mpl.colors.TwoSlopeNorm(vmin=0, vmax=max_contrast, vcenter=self._low_contrast_threshold)

        # add to graph
        num_vertices = 32
        for roi in self.low_contrast_rois:
            xs, ys = [], []
            for i in range(num_vertices + 1):
                angle = 2 * np.pi * i / num_vertices
                xs.append(roi.center.x + roi.radius * math.cos(angle))
                ys.append(roi.center.y + roi.radius * math.sin(angle))
            c = f'rgba{cmap(norm(roi.contrast))}'
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                fill='toself',
                fillcolor=c,
                mode='none',
                showlegend=True if roi == self.low_contrast_rois[0] else False,
                legendgroup='Low Contrast',
                legendgrouptitle={'text': 'Regions of Interest'},
                text=f'{roi.contrast:.3f}',
                name='Contrast',
            ))

        # Add high contrast ROIs
        for roi in self.high_contrast_rois:
            svg_path = (f'M {roi.pt_btm_left_px.x}, {roi.pt_btm_left_px.y} ' +
                        f'L {roi.pt_top_left_px.x}, {roi.pt_top_left_px.y} ' +
                        f'L {roi.pt_top_right_px.x}, {roi.pt_top_right_px.y} ' +
                        f'L {roi.pt_btm_right_px.x}, {roi.pt_btm_right_px.y} ' +
                        f'Z')
            fig.add_shape(
                type='path',
                path=svg_path,
                line_color='Yellow',
                showlegend=True,
                name='HighContrast ROIs',
                legendgroup='HighContrast ROIs',
            )

        # Display figure if requested
        if show:
            fig.show()

        return fig