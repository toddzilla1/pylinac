import math
import random
import typing
import warnings

import plotly.express as px
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import py_linq
import pydantic

import pylinac
import pylinac.core.roi

import ecu.pylinac_utils.roi


class SampledProfileMTF(pylinac.core.mtf.MTF):
    """Calculates the modulation transfer function (MTF) of a sampled profile."""

    def __init__(self, rois: typing.List[ecu.pylinac_utils.roi.HighContrastRectangularROI]):
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
    _roi_r1 = 0.775
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

    # Overriding base method contrast methods
    high_contrast_roi_settings = None

    high_contrast_roi_strips = [
        ecu.pylinac_utils.roi.HighContrastRectangularROIStrip.define_relative_to_phantom(
            location_radius_rel_phantom=0.234,
            location_angle_deg_rel_phantom=90,
            width_rel_phantom=0.108,
            height_rel_phantom=0.48,
            rotation_deg_rel_phantom=180.0,
        ),
        ecu.pylinac_utils.roi.HighContrastRectangularROIStrip.define_relative_to_phantom(
            location_radius_rel_phantom=0.076,
            location_angle_deg_rel_phantom=90,
            width_rel_phantom=0.108,
            height_rel_phantom=0.48,
            rotation_deg_rel_phantom=180.0,
        ),
        ecu.pylinac_utils.roi.HighContrastRectangularROIStrip.define_relative_to_phantom(
            location_radius_rel_phantom=0.08,
            location_angle_deg_rel_phantom=270,
            width_rel_phantom=0.108,
            height_rel_phantom=0.48,
            rotation_deg_rel_phantom=180.0,
        ),
    ]

    # fractional divisions measured relative to profile length at (quantile(0.80) + quantile(0.02)) / 2
    high_contrast_roi_strips[0].all_nominal_line_pairs_per_mm = [0.5, 0.71, 1.0, 1.4, 2.0, 2.8, 4.0]
    high_contrast_roi_strips[0].all_nominal_line_pairs_per_region = [5, 5, 5, 5, 5, 5, 5]
    high_contrast_roi_strips[0].region_fractional_divisions = [-0.027962, 0.294, 0.52885, 0.6983, 0.81776, 0.90176,
                                                               0.96164, 1.0038]

    high_contrast_roi_strips[1].all_nominal_line_pairs_per_mm = [0.56, 0.8, 1.12, 1.6, 2.24, 3.15, 4.5]
    high_contrast_roi_strips[1].all_nominal_line_pairs_per_region = [5, 5, 5, 5, 5, 5, 5]
    high_contrast_roi_strips[1].region_fractional_divisions = [-0.02553, 0.28228, 0.52156, 0.69467, 0.81519, 0.90145,
                                                               0.9623, 1.009]

    high_contrast_roi_strips[2].all_nominal_line_pairs_per_mm = [0.63, 0.9, 1.25, 1.8, 2.5, 3.55, 5.0]
    high_contrast_roi_strips[2].all_nominal_line_pairs_per_region = [5, 5, 5, 5, 5, 5, 5]
    high_contrast_roi_strips[2].region_fractional_divisions = [-0.039873, 0.27358, 0.51645, 0.69112, 0.81195, 0.89989,
                                                               0.9618, 1.01]

    high_contrast_max_roi = ecu.pylinac_utils.roi.RectangularROI.define_relative_to_phantom(
        location_radius_rel_phantom=0.255,
        location_angle_deg_rel_phantom=271,
        width_rel_phantom=0.09,
        height_rel_phantom=0.09,
        rotation_deg_rel_phantom=0.0,
    )

    high_contrast_min_roi = ecu.pylinac_utils.roi.RectangularROI.define_relative_to_phantom(
        location_radius_rel_phantom=0.302,
        location_angle_deg_rel_phantom=303,
        width_rel_phantom=0.09,
        height_rel_phantom=0.09,
        rotation_deg_rel_phantom=0.0,
    )

    high_contrast_rectangular_rois: typing.List[ecu.pylinac_utils.roi.HighContrastRectangularROI] = []

    # Override to return rectangular ROIs instead of disc
    def _sample_high_contrast_rois(self) -> list[ecu.pylinac_utils.roi.HighContrastRectangularROI]:
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

        # place min / max regions on image
        pixel_size_mm = 1.0 / self.image.dpmm
        self.high_contrast_min_roi.place_roi_on_image(
            image_array=self.image.array,
            phantom_center_px=self.high_res_center,
            phantom_angle_deg_rel_image=self.phantom_angle,
            phantom_radius_px=self.phantom_radius,
            pixel_size_mm=pixel_size_mm,
        )

        self.high_contrast_max_roi.place_roi_on_image(
            image_array=self.image.array,
            phantom_center_px=self.high_res_center,
            phantom_angle_deg_rel_image=self.phantom_angle,
            phantom_radius_px=self.phantom_radius,
            pixel_size_mm=pixel_size_mm,
        )

        # OVERRIDDEN to use custom high contrast ROIs strips
        hc_rois:typing.List[ecu.pylinac_utils.roi.HighContrastRectangularROI] = []
        strip: ecu.pylinac_utils.roi.HighContrastRectangularROIStrip
        for strip in self.high_contrast_roi_strips:
            strip.expected_max_pixel_value = self.high_contrast_max_roi.mean
            strip.expected_min_pixel_value = self.high_contrast_min_roi.mean
            strip.place_roi_on_image(
                image_array=self.image.array,
                phantom_center_px=self.high_res_center,
                phantom_angle_deg_rel_image=self.phantom_angle,
                phantom_radius_px=self.phantom_radius,
                pixel_size_mm=pixel_size_mm,
            )

            for roi in strip.regions:
                hc_rois.append(roi)

            #todo process strip into ROIs

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
            x_adjustment: float = 0,
            y_adjustment: float = 0,
            angle_adjustment: float = 0,
            roi_size_factor: float = 1,
            scaling_factor: float = 1,
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
        self.high_contrast_rectangular_rois = self._sample_high_contrast_rois()

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

            lc_roi = pylinac.core.roi.LowContrastDiskROI.from_phantom_center(
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

    @pydantic.validate_arguments
    def plot_interactive_rois(self, colors: list = px.colors.sequential.gray, color_cycles: pydantic.PositiveInt = 1,
                              image_only: bool = False, show: bool = True) -> go.Figure:
        """ Create interactive figure of the ROIs for Leeds TOR analysis. """

        #
        # Background image
        #

        fig = px.imshow(
            img=self.image.array,
            color_continuous_scale=colors * color_cycles,
            height=self.image.array.shape[0],
            width=self.image.array.shape[1],
        )

        # No colorbar
        fig.update_layout(coloraxis_showscale=False)

        # Phantom outline
        if not image_only:
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=self.phantom_center.x - self.phantom_radius,
                y0=self.phantom_center.y - self.phantom_radius,
                x1=self.phantom_center.x + self.phantom_radius,
                y1=self.phantom_center.y + self.phantom_radius,
                line_color="Blue",
                showlegend=True,
                legendgroup='Phantom',
                legendgrouptitle={'text': 'Phantom'},
                name='Outline',
                legendrank=1,
            )

            # Phantom center
            fig.add_trace(go.Scatter(
                mode='markers',
                x=[self.phantom_center.x],
                y=[self.phantom_center.y],
                marker=dict(symbol='cross-thin', line_color='blue', size=12, line_width=2),
                showlegend=True,
                legendgroup='Phantom',
                legendgrouptitle={'text': 'Phantom'},
                name='Center',
                legendrank=1,
            ))

            # High contrast phantom center
            fig.add_trace(go.Scatter(
                mode='markers',
                x=[self.high_res_center.x],
                y=[self.high_res_center.y],
                marker=dict(symbol='x-thin', line_color='blue', size=12, line_width=2),
                showlegend=True,
                legendgroup='Phantom',
                legendgrouptitle={'text': 'Phantom'},
                name='HC Center',
                legendrank=1,
            ))

        #
        # Add low contrast ROIs
        #

        if not image_only:
            # find max contrast
            max_contrast = np.max([roi.contrast for roi in self.low_contrast_rois])

            # define color normalization (borrow from matplotlib)
            norm = mpl.colors.TwoSlopeNorm(vmin=0, vmax=float(max_contrast), vcenter=float(self._low_contrast_threshold))

            # add to graph
            num_vertices = 32
            for roi in self.low_contrast_rois:
                xs, ys = [], []
                for i in range(num_vertices + 1):
                    angle = 2 * np.pi * i / num_vertices
                    xs.append(roi.center.x + roi.radius * math.cos(angle))
                    ys.append(roi.center.y + roi.radius * math.sin(angle))
                c = px.colors.sample_colorscale(px.colors.diverging.RdYlGn, norm(roi.contrast))[0]
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    fill='toself',
                    fillcolor=c,
                    mode='none',
                    showlegend=True, #if roi == self.low_contrast_rois[0] else False,
                    legendgroup='lc_rois',
                    legendgrouptitle={'text': 'Low Contrast'},
                    text=f'{roi.contrast:.3f}',
                    name=f'{roi.contrast:.3f}',
                    legendrank=2,
                ))

        # Add high contrast ROIs
        if not image_only:
            self.high_contrast_max_roi.add_to_interactive_plot(
                fig=fig,
                line_color='Yellow',
                opacity=0.4,
                name='Max',
                show_legend=True,
                legend_group_title='High Contrast',
                show_arrow=False,
            )

            self.high_contrast_min_roi.add_to_interactive_plot(
                fig=fig,
                line_color='Yellow',
                opacity=0.4,
                name='Min',
                show_legend=True,
                legend_group_title='High Contrast',
                show_arrow=False,
            )

            strip: ecu.pylinac_utils.roi.HighContrastRectangularROIStrip
            for i, strip in enumerate(self.high_contrast_roi_strips, start=1):
                strip.add_to_interactive_plot(
                    fig=fig,
                    line_color='Green',
                    opacity=0.4,
                    name=f'Strip {i}',
                    show_legend=True,
                    legend_group_title='High Contrast',
                )

            roi: ecu.pylinac_utils.roi.HighContrastRectangularROI
            for roi in self.high_contrast_rectangular_rois:
                roi.add_to_interactive_plot(
                    fig=fig,
                    line_color='Yellow',
                    opacity=0.4,
                    name=f'{roi.nominal_line_pairs_per_mm}',
                    show_legend=True,
                    legend_group_title='High Contrast',
                    show_arrow=False,
                )

        # update group click action
        fig.update_layout(legend=dict(groupclick="toggleitem"))

        # configure plot options
        config = {
            # 'scrollZoom': True,
            'displaylogo': False,
            # 'responsive': True,
            'toImageButtonOptions': {
                'filename': 'leeds_analysis'
            }
        }

        # Display figure if requested
        if show:
            fig.show(config=config)

        # fig.write_html('/Users/zilla/Downloads/leeds_analysis.html', config=config)
        return fig
