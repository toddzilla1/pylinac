import math

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import scipy.interpolate

import ecu.pylinac_utils.leeds
import pylinac
import numpy as np
import pydantic
import pydantic_numpy
import typing
import functools
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches

# typing.Self not defined until Python 3.11. Work-around in the meantime.
Self = typing.TypeVar('Self', bound='AnnulusROI')


# Used to clear properties cached by @functools.cached_property
def clear_cached_property(instance, property_name):
    if property_name in instance.__dict__:
        del instance.__dict__[property_name]


class LowContrastDiscROI(pylinac.core.roi.LowContrastDiskROI):
    """Override to implement custom plotting."""

    def add_to_interactive_plot(
            self,
            fig: go.Figure,
            color: str = 'White',
            show_legend: bool = True,
            legend_group: str = 'lc_rois',
            legend_group_title: str = 'Low Contrast',
            legend_group_rank: int = None,
            legend_text: str = 'ROI',
    ):
        xs, ys = [], []
        num_vertices = 32
        for i in range(num_vertices + 1):
            angle = 2 * np.pi * i / num_vertices
            xs.append(self.center.x + self.radius * math.cos(angle))
            ys.append(self.center.y + self.radius * math.sin(angle))
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            fill='toself',
            fillcolor=color,
            mode='none',
            showlegend=show_legend,
            legendgroup=legend_group,
            legendgrouptitle={'text': legend_group_title},
            legendrank=legend_group_rank,
            text=f'{self.contrast:.3f}',
            name=legend_text,
        ))


class AnnulusROI(pydantic.BaseModel):
    """A class representing an annulus-shaped Region of Interest."""

    image_array: pydantic_numpy.typing.NpNDArray = pydantic.Field(frozen=True)
    angle_deg: float = pydantic.Field(frozen=True)
    dist_from_center_px: float = pydantic.Field(frozen=True)
    roi_outer_radius_px: float = pydantic.Field(gt=0, frozen=True)
    roi_inner_radius_px: float = pydantic.Field(ge=0, frozen=True)
    # noinspection PyUnresolvedReferences
    phantom_center_pt_px: pylinac.core.geometry.Point = pydantic.Field(frozen=True)

    @pydantic.model_validator(mode='after')
    def _check_roi_inner_outer_radius_px(self) -> Self:
        if self.roi_outer_radius_px <= self.roi_inner_radius_px:
            raise ValueError(
                'Outer radius cannot be less than or equal to inner radius.'
            )
        return self

    class Config:
        arbitrary_types_allowed = True

    @functools.cached_property
    def center_pt_px(self) -> pylinac.core.geometry.Point:
        y_shift = np.sin(np.deg2rad(self.angle_deg)) * self.dist_from_center_px
        x_shift = np.cos(np.deg2rad(self.angle_deg)) * self.dist_from_center_px
        return pylinac.geometry.Point(
            x=self.phantom_center_pt_px.x + x_shift,
            y=self.phantom_center_pt_px.y + y_shift
        )

    @functools.cached_property
    def masked_image(self) -> np.ndarray:
        """Return a mask of the image, only exposing the annular ROI."""
        mask = np.ones(shape=self.image_array.shape)
        mask[skimage.draw.disk(
            center=[self.center_pt_px.y, self.center_pt_px.x],
            radius=self.roi_outer_radius_px)] = 0
        mask[skimage.draw.disk(
            center=[self.center_pt_px.y, self.center_pt_px.x],
            radius=self.roi_inner_radius_px)] = 1
        return np.ma.MaskedArray(data=self.image_array, mask=mask)

    @functools.cached_property
    def pixel_values(self) -> np.ndarray:
        return self.masked_image

    @functools.cached_property
    def pixel_value(self) -> float:
        """The median pixel value of the ROI."""
        return float(np.ma.median(self.masked_image))

    @functools.cached_property
    def std(self) -> float:
        """The standard deviation of the pixel values."""
        return float(np.ma.std(self.masked_image))

    @functools.cached_property
    def area(self) -> float:
        """The area of the ROI."""
        return ((np.pi * self.roi_outer_radius_px * self.roi_outer_radius_px) -
                (np.pi * self.roi_inner_radius_px * self.roi_inner_radius_px))

    def plot2axes(
            self,
            axes: plt.Axes | None = None,
            edgecolor: str = "black",
            fill: bool = False,
            text: str = "",
            fontsize: str = "medium",
            **kwargs,
    ) -> None:
        """Plot the Annulus on the axes.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            An MPL axes to plot to.
        edgecolor : str
            The color of the annulus.
        fill : bool
            Whether to fill the annulus with color or leave hollow.
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
            matplotlib.patches.Annulus(
                xy=(self.center_pt_px.x, self.center_pt_px.y),
                r=self.roi_outer_radius_px,
                width=self.roi_outer_radius_px - self.roi_inner_radius_px,
                edgecolor=edgecolor,
                fill=fill,
                **kwargs,
            )
        )

        if text:
            axes.text(
                x=self.center_pt_px.x,
                y=self.center_pt_px.y,
                s=text,
                fontsize=fontsize,
                color=edgecolor,
            )

    def get_outline_coordinates_for_plot(self, num_vertices: int = 64) -> typing.Tuple[
        typing.Tuple[typing.List, typing.List], typing.Tuple[typing.List, typing.List]]:
        """
        Provides coordinates for drawing the ROI outline on a plot. An annulus is typically drawn in plotly as two
        separate traces, so the inner and outer coordinates are returned separately.

        Returns:
            (outer_xs, outer_ys), (inner_xs, inner_ys)
        """

        def calc_outline(radius: float) -> typing.Tuple[typing.List, typing.List]:
            xs, ys = [], []
            for i in range(num_vertices + 1):
                angle = 2 * np.pi * i / num_vertices
                xs.append(self.center_pt_px.x + radius * math.cos(angle))
                ys.append(self.center_pt_px.y + radius * math.sin(angle))
            return xs, ys

        return calc_outline(self.roi_outer_radius_px), calc_outline(self.roi_inner_radius_px)

    def add_to_interactive_plot(
            self,
            fig: go.Figure,
            color: str = 'White',
            show_legend: bool = True,
            legend_group: str = 'lc_rois',
            legend_group_title: str = 'Low Contrast',
            legend_group_rank: int = None,
            hover_text: str = None,
            legend_text: str = None,
    ):
        """Add ROI to plotly figure"""
        (outer_xs, outer_ys), (inner_xs, inner_ys) = self.get_outline_coordinates_for_plot()

        # draw annulus 'hole' first
        fig.add_trace(go.Scatter(
            x=inner_xs,
            y=inner_ys,
            fill='none',
            # fillcolor='rgba(0,0,0,0)',
            mode='none',
            showlegend=False,
            legendgroup=legend_group,
            legendgrouptitle={'text': legend_group_title},
            legendrank=legend_group_rank,
            text=hover_text,
            name=legend_text,
        ))

        fig.add_trace(go.Scatter(
            x=outer_xs,
            y=outer_ys,
            fill='tonext',
            fillcolor=color,
            mode='none',
            showlegend=show_legend,
            legendgroup=legend_group,
            legendgrouptitle={'text': legend_group_title},
            legendrank=legend_group_rank,
            text=hover_text,
            name=legend_text,
        ))

    def as_dict(self) -> dict:
        """Convert to dict. Useful for dataclasses/Result"""
        return {
            "center_x": self.center_pt_px.x,
            "center_y": self.center_pt_px.y,
            "outer radius": self.roi_outer_radius_px,
            "inner radius": self.roi_inner_radius_px,
            "median": self.pixel_value,
            "std": self.std,
        }


class RectangularROI(pydantic.BaseModel):
    """A rectangular-shaped region of interest, possibly rotated relative to the x/y axes."""

    location_radius_rel_phantom: float = pydantic.Field(
        default=None,
        description='Distance from phantom center to ROI center, relative to phantom radius.'
    )

    location_angle_deg_rel_phantom: float = pydantic.Field(
        default=None,
        description='Clockwise angle, in degrees, of vector pointing to ROI center relative to phantom orientation.'
    )

    width_rel_phantom: float = pydantic.Field(
        default=None,
        description='ROI width relative to phantom radius.'
    )

    height_rel_phantom: float = pydantic.Field(
        default=None,
        description='ROI height relative to phantom radius.'
    )

    rotation_deg_rel_phantom: float = pydantic.Field(
        default=None,
        description='ROI rotation around its center relative to phantom orientation.'
    )

    rotation_rad_rel_image: typing.Optional[float] = pydantic.Field(
        default=None,
        description='ROI rotation around its center relative to image axes.'
    )

    # noinspection PyUnresolvedReferences
    phantom_center_px: typing.Optional[pylinac.core.geometry.Point] = pydantic.Field(
        default=None,
        description='Phantom center in image pixels'
    )

    # noinspection PyUnresolvedReferences
    center_pt_px: typing.Optional[pylinac.core.geometry.Point] = pydantic.Field(
        default=None,
        description='ROI center in image pixels'
    )

    image_array: typing.Optional[pydantic_numpy.typing.NpNDArray] = pydantic.Field(
        default=None,
        description='Array containing phantom image'
    )

    width_px: typing.Optional[float] = pydantic.Field(
        default=None,
        description='ROI width in image pixels'
    )

    height_px: typing.Optional[float] = pydantic.Field(
        default=None,
        description='ROI height in image pixels'
    )

    location_angle_rad_rel_image: typing.Optional[float] = pydantic.Field(
        default=None,
        description='Clockwise angle, in radians, of vector pointing to ROI center relative to image axes.'
    )

    pixel_size_mm: typing.Optional[float] = pydantic.Field(
        default=None,
        description='Pixel size in mm. Assumes square pixels.'
    )

    # Constants for labeling Pandas columns; used in both HighContrastRectangularROI and HighContrastRectangularROIStrip
    # so placed here to be visible to both.
    COL_PIXEL_VALUE: typing.ClassVar[str] = 'Pixel Value'
    COL_DISTANCE_PX: typing.ClassVar[str] = 'Distance (Pixels)'
    COL_DISTANCE_MM: typing.ClassVar[str] = 'Distance (mm)'
    COL_DATA_TYPE: typing.ClassVar[str] = 'Data'
    COL_HANNING: typing.ClassVar[str] = 'Hanning'
    COL_FFT_MAGS: typing.ClassVar[str] = 'FFT Magnitude'
    COL_FFT_FREQS: typing.ClassVar[str] = 'FFT Frequency (1/mm)'

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def define_relative_to_phantom(
            cls,
            location_radius_rel_phantom: float,
            location_angle_deg_rel_phantom: float,
            width_rel_phantom: float,
            height_rel_phantom: float,
            rotation_deg_rel_phantom: float) -> 'RectangularROI':

        return RectangularROI(
            location_radius_rel_phantom=location_radius_rel_phantom,
            location_angle_deg_rel_phantom=location_angle_deg_rel_phantom,
            width_rel_phantom=width_rel_phantom,
            height_rel_phantom=height_rel_phantom,
            rotation_deg_rel_phantom=rotation_deg_rel_phantom,
        )

    # Used to find ROI pixel center and size once phantom pixels, center, radius, and angle are known
    # noinspection PyUnresolvedReferences
    def place_roi_on_image(self, image_array: pydantic_numpy.typing.NpNDArray,
                           phantom_center_px: pylinac.core.geometry.Point,
                           phantom_angle_deg_rel_image: float, phantom_radius_px: float, pixel_size_mm: float):

        self.image_array = image_array
        self.pixel_size_mm = pixel_size_mm

        # final location angle = phantom + ROI angles
        self.location_angle_rad_rel_image = np.deg2rad(
            phantom_angle_deg_rel_image + self.location_angle_deg_rel_phantom)

        # final orientation angle = phantom + ROI
        self.rotation_rad_rel_image = np.deg2rad(phantom_angle_deg_rel_image + self.rotation_deg_rel_phantom)

        # ROI distance from center is relative to phantom radius
        radius_px = phantom_radius_px * self.location_radius_rel_phantom

        # calc ROI center in image
        self.center_pt_px = pylinac.core.geometry.Point(
            x=phantom_center_px.x + np.cos(self.location_angle_rad_rel_image) * radius_px,
            y=phantom_center_px.y + np.sin(self.location_angle_rad_rel_image) * radius_px,
        )

        # determine ROI width and height in pixels
        self.width_px = phantom_radius_px * self.width_rel_phantom
        self.height_px = phantom_radius_px * self.height_rel_phantom

        # clear any cached properties
        self.clear_cached_properties()

    def clear_cached_properties(self):
        clear_cached_property(self, '_delta_x1')
        clear_cached_property(self, '_delta_x2')
        clear_cached_property(self, '_delta_y1')
        clear_cached_property(self, '_delta_y2')
        clear_cached_property(self, 'pt_btm_middle_px')
        clear_cached_property(self, 'pt_btm_left_px')
        clear_cached_property(self, 'pt_btm_right_px')
        clear_cached_property(self, 'pt_top_middle_px')
        clear_cached_property(self, 'pt_top_left_px')
        clear_cached_property(self, 'pt_top_right_px')
        clear_cached_property(self, 'masked_pixel_values')
        clear_cached_property(self, 'masked_pixel_coordinates')
        clear_cached_property(self, 'mean')
        clear_cached_property(self, 'std')

    @functools.cached_property
    def _delta_x1(self) -> float:
        return (self.height_px / 2.0) * math.cos(self.rotation_rad_rel_image)

    @functools.cached_property
    def _delta_y1(self) -> float:
        return (self.height_px / 2.0) * math.sin(self.rotation_rad_rel_image)

    @functools.cached_property
    def _delta_x2(self) -> float:
        return (self.width_px / 2.0) * math.sin(self.rotation_rad_rel_image)

    @functools.cached_property
    def _delta_y2(self) -> float:
        return (self.width_px / 2.0) * math.cos(self.rotation_rad_rel_image)

    # noinspection PyUnresolvedReferences
    @functools.cached_property
    def pt_btm_middle_px(self) -> pylinac.core.geometry.Point:
        return pylinac.core.geometry.Point(
            self.center_pt_px.x - self._delta_x1,
            self.center_pt_px.y - self._delta_y1
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
            self.center_pt_px.x + self._delta_x1,
            self.center_pt_px.y + self._delta_y1
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
    def masked_pixel_coordinates(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
        - A tuple (rr, cc) where:
            - rr: Array of row indices for the pixels inside the polygon.
            - cc: Array of column indices for the pixels inside the polygon.
        """
        rr, cc = skimage.draw.polygon(
            [self.pt_btm_left_px.y, self.pt_top_left_px.y, self.pt_top_right_px.y, self.pt_btm_right_px.y],
            [self.pt_btm_left_px.x, self.pt_top_left_px.x, self.pt_top_right_px.x, self.pt_btm_right_px.x]
        )

        return rr, cc

    @functools.cached_property
    def masked_pixel_values(self) -> np.ndarray:
        """
        Returns:
            - A 1D numpy array of pixel values from the ROI.
        """
        rr, cc = self.masked_pixel_coordinates
        return self.image_array[rr, cc]

    @functools.cached_property
    def mean(self) -> np.floating:
        return np.mean(self.masked_pixel_values)

    @functools.cached_property
    def std(self) -> np.floating:
        return np.std(self.masked_pixel_values)

    def add_to_interactive_plot(
            self,
            fig: plotly.graph_objs.Figure,
            line_color: str = None,
            fill_color: str = None,
            opacity: float = 1.0,
            hover_text: str = None,
            legend_text: str = 'ROI',
            show_legend: bool = True,
            legend_group: str = None,
            legend_group_title: str = None,
            legend_group_rank: int = None,
            show_arrow: bool = True,
    ) -> None:
        """Add this ROI to the interactive plot."""

        # get coordinates for ROI outline
        data = [
            list(self.pt_btm_left_px.as_array()),
            list(self.pt_top_left_px.as_array()),
            list(self.pt_top_right_px.as_array()),
            list(self.pt_btm_right_px.as_array()),
            list(self.pt_btm_left_px.as_array()),
        ]

        if show_arrow:
            arrow_delta = self.width_px if self.width_px < self.height_px else self.height_px
            arrow_delta *= 0.25
            arrow_angle_rad = np.deg2rad(45.0)
            # noinspection PyUnresolvedReferences
            arrow_pt1 = pylinac.core.geometry.Point(
                self.center_pt_px.x - arrow_delta * math.cos(self.rotation_rad_rel_image + arrow_angle_rad),
                self.center_pt_px.y - arrow_delta * math.sin(self.rotation_rad_rel_image + arrow_angle_rad),
            )
            # noinspection PyUnresolvedReferences
            arrow_pt2 = pylinac.core.geometry.Point(
                self.center_pt_px.x - arrow_delta * math.cos(self.rotation_rad_rel_image - arrow_angle_rad),
                self.center_pt_px.y - arrow_delta * math.sin(self.rotation_rad_rel_image - arrow_angle_rad),
            )

            data += [[math.nan, math.nan]]
            data += [list(self.pt_btm_middle_px.as_array())]
            data += [list(self.center_pt_px.as_array())]
            data += [list(arrow_pt1.as_array())]
            data += [list(arrow_pt2.as_array())]
            data += [list(self.center_pt_px.as_array())]

        # extract data
        xs = [pt[0] for pt in data]
        ys = [pt[1] for pt in data]

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            fill='none' if fill_color is None else 'toself',
            fillcolor=fill_color,
            opacity=opacity,
            mode='none' if line_color is None else 'lines',
            line=dict(color=line_color),
            showlegend=show_legend,
            legendgroup=legend_group,
            legendgrouptitle=dict(text=legend_group_title),
            legendrank=legend_group_rank,
            text=f'mean = {self.mean:.3f}' if hover_text is None else hover_text,
            name=legend_text,
        ))

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
                x=self.center_pt_px.x,
                y=self.center_pt_px.y,
                s=text,
                fontsize=fontsize,
                color=edgecolor,
            )


class HighContrastRectangularROI(RectangularROI):
    """Rectangular region of interest for a high contrast bar pattern in image."""

    nominal_line_pairs: float = pydantic.Field(
        frozen=True,
        description='The expected number of line pairs in this region.'
    )

    nominal_line_pairs_per_mm: float = pydantic.Field(
        frozen=True,
        description='The expected line pairs per mm for this high contrast bar pattern',
    )

    measured_line_pairs_per_mm: float = pydantic.Field(
        default=None,
        description='The measured line pairs per mm for this high contrast bar pattern',
    )

    nominal_max: float = pydantic.Field(
        frozen=True,
        description='The expected maximum pixel value for an ideal imaging system.',
    )

    nominal_min: float = pydantic.Field(
        frozen=True,
        description='The expected minimum pixel value for an ideal imaging system.',
    )

    profile_resampled: pd.DataFrame = pydantic.Field(
        frozen=True,
        description='The profile resampled at equal intervals to facilitate FFT calculation',
    )

    profile_raw: pd.DataFrame = pydantic.Field(
        frozen=True,
        description='Raw profile data as collected from the image.',
    )

    resampling_interval_mm: float = pydantic.Field(
        frozen=True,
        description='Distance between resampled profile data points, in mm',
    )

    fft_data: pd.DataFrame = pydantic.Field(
        default=None,
        description='FFT data calculated during calculation of the MTF'
    )

    fft_data_interpolated: pd.DataFrame = pydantic.Field(
        default=None,
        description='FFT data interpolated by cubic spline'
    )

    # MTF = 0 if measured frequency varies by more than this amount from the nominal frequency
    ALLOWED_FREQUENCY_DETECTION_ERROR: typing.ClassVar[float] = 0.10

    @classmethod
    def from_strip_data(cls,
                        nominal_line_pairs: float,
                        nominal_line_pairs_per_mm: float,
                        nominal_max: float,
                        nominal_min: float,
                        center_pt_px: pylinac.core.geometry.Point,
                        width_px: float,
                        height_px: float,
                        image_array: np.ndarray,
                        pixel_size_mm: float,
                        rotation_rad_rel_image: float,
                        profile_raw: pd.DataFrame,
                        profile_resampled: pd.DataFrame,
                        resampling_interval_mm: float,
                        ) -> 'HighContrastRectangularROI':
        return HighContrastRectangularROI(
            nominal_line_pairs=nominal_line_pairs,
            nominal_line_pairs_per_mm=nominal_line_pairs_per_mm,
            nominal_max=nominal_max,
            nominal_min=nominal_min,
            center_pt_px=center_pt_px,
            width_px=width_px,
            height_px=height_px,
            image_array=image_array,
            pixel_size_mm=pixel_size_mm,
            rotation_rad_rel_image=rotation_rad_rel_image,
            profile_resampled=profile_resampled,
            profile_raw=profile_raw,
            resampling_interval_mm=resampling_interval_mm,
        )

    @classmethod
    def find_profile_beg_and_end(cls,
                                 df: pd.DataFrame,
                                 target_pixel_value: float,
                                 smoothing: pydantic.PositiveInt = 5) -> typing.Tuple[int, int]:
        """
        Finds the first and last occurrence of the profile passing above a target pixel value, and then goes back
        (or forward for end) until the profile stops decreasing. Returns those positions and indexes.

        Returns:
            left edge dataframe index label
            right edge dataframe index label
        """

        # Create a df from values above the threshold (maintains original indexes)
        df_above = df[df[cls.COL_PIXEL_VALUE] > target_pixel_value]

        # Beginning crossing index label is first entry
        index_lbl_beg = df_above.index[0]

        # End crossing index is the last entry
        index_lbl_end = df_above.index[-1]

        # Rewind beginning until profile stops decreasing
        edge_lbl = df[cls.COL_PIXEL_VALUE].index[0] + 2 * smoothing - 1
        if index_lbl_beg < edge_lbl:  # sanity check
            index_lbl_beg = edge_lbl
        for index_lbl_beg in range(index_lbl_beg, edge_lbl, -1):
            here_value = df.loc[index_lbl_beg - smoothing + 1:index_lbl_beg, cls.COL_PIXEL_VALUE].mean()
            there_value = df.loc[index_lbl_beg - 2 * smoothing + 1:index_lbl_beg - smoothing,
                          cls.COL_PIXEL_VALUE].mean()
            if there_value >= here_value:
                break
        index_lbl_beg = index_lbl_beg - smoothing + 1

        # Move forward to end until profile stops decreasing
        edge_lbl = df[cls.COL_PIXEL_VALUE].index[-1] - 2 * smoothing + 1
        if index_lbl_end > edge_lbl:  # sanity check
            index_lbl_end = edge_lbl
        for index_lbl_end in range(index_lbl_end, edge_lbl):
            here_value = df.loc[index_lbl_end:index_lbl_end + smoothing - 1, cls.COL_PIXEL_VALUE].mean()
            there_value = df.loc[index_lbl_end + smoothing:index_lbl_end + 2 * smoothing - 1,
                          cls.COL_PIXEL_VALUE].mean()
            if there_value >= here_value:
                break
        index_lbl_end = index_lbl_end + smoothing - 1

        return index_lbl_beg, index_lbl_end

    @functools.cached_property
    def ideal_profile_square_wave(self) -> pd.DataFrame:
        """Dataframe with points representing the ideal square wave profile"""

        # Create new df with same x values as this ROI
        df = self.profile_resampled[[self.COL_DISTANCE_MM]].copy()

        # Figure out wavelength
        wavelength_mm = 1.0 / self.nominal_line_pairs_per_mm

        # Figure out phase shift by finding first mid-line crossing
        top = self.profile_resampled[self.COL_PIXEL_VALUE].quantile(0.98)
        bottom = self.profile_resampled[self.COL_PIXEL_VALUE].quantile(0.02)
        middle = (top + bottom) / 2
        start_index = (self.profile_resampled[self.COL_PIXEL_VALUE] > middle).idxmax()
        phase_shift_mm = self.profile_resampled.loc[start_index, self.COL_DISTANCE_MM]

        # Figure out amplitude
        amplitude = (self.nominal_max - self.nominal_min) / 2.0

        # Calculate ideal pixel values
        df.loc[:, self.COL_PIXEL_VALUE] = ((1 + scipy.signal.square((df[self.COL_DISTANCE_MM] - phase_shift_mm) * 2 *
                                                                    np.pi / wavelength_mm)) * amplitude +
                                           self.nominal_min)
        df.loc[:, self.COL_DATA_TYPE] = 'Ideal'

        return df

    @functools.cached_property
    def ideal_profile_sine_wave(self) -> pd.DataFrame:
        """Dataframe with points representing the ideal sine wave profile"""

        # Create new df with same x values as this ROI
        df = self.profile_resampled[[self.COL_DISTANCE_MM]].copy()

        # Figure out wavelength
        wavelength_mm = 1.0 / self.nominal_line_pairs_per_mm

        # Figure out phase shift by finding first mid-line crossing
        top = self.profile_resampled[self.COL_PIXEL_VALUE].quantile(0.98)
        bottom = self.profile_resampled[self.COL_PIXEL_VALUE].quantile(0.02)
        middle = (top + bottom) / 2
        start_index = (self.profile_resampled[self.COL_PIXEL_VALUE] > middle).idxmax()
        phase_shift_mm = self.profile_resampled.loc[start_index, self.COL_DISTANCE_MM]

        # Figure out amplitude
        amplitude = (self.nominal_max - self.nominal_min) / 2.0

        # Calculate ideal pixel values
        ys = [amplitude * (1 + math.sin((x - phase_shift_mm) * 2 * math.pi / wavelength_mm)) + self.nominal_min for x in
              self.profile_resampled[self.COL_DISTANCE_MM].values]
        df.loc[:, self.COL_PIXEL_VALUE] = ys
        df.loc[:, self.COL_DATA_TYPE] = 'Ideal'

        return df

    @functools.cached_property
    def ideal_fft_max_magnitude(self) -> float:
        """Calculates the maximum magnitude of an FFT for the ideal sin wave profile at the nominal frequency"""

        # Get ideal profile
        df = self.ideal_profile_sine_wave

        # Apply Hanning window to remove artifacts created from chopping the waveform at left and right edges
        df.loc[:, self.COL_HANNING] = (df[self.COL_PIXEL_VALUE] * np.hanning(len(df[self.COL_PIXEL_VALUE])))

        # Calculate FFT magnitudes
        fft_magnitudes = np.abs(np.fft.fft(df[self.COL_HANNING]))

        # Calculate FFT frequencies
        num_samples = len(fft_magnitudes)
        fft_frequencies = np.fft.fftfreq(n=num_samples, d=self.resampling_interval_mm)

        # Create a dataframe with the FFT data. Only keep positive frequencies above approximately DC (0 lp/mm)
        fft_data = pd.DataFrame({
            self.COL_FFT_MAGS: fft_magnitudes[2:num_samples // 2],
            self.COL_FFT_FREQS: fft_frequencies[2:num_samples // 2],
        })

        # Create a monotonic (doesn't 'overshoot') cubic spline to interpolate between discrete FFT results
        # noinspection PyUnresolvedReferences
        spline = scipy.interpolate.InterpolatedUnivariateSpline(
            x=fft_data[self.COL_FFT_FREQS],
            y=fft_data[self.COL_FFT_MAGS],
            ext='zeros',
        )

        # Return FFT magnitude at ideal frequency
        return spline(self.nominal_line_pairs_per_mm)

    @functools.cached_property
    def mtf(self) -> float:
        """Modulation Transfer Function for the ROI"""

        # Apply Hanning window to remove artifacts created from chopping the waveform at left and right edges
        self.profile_resampled.loc[:, self.COL_HANNING] = (
                self.profile_resampled[self.COL_PIXEL_VALUE] *
                np.hanning(len(self.profile_resampled[self.COL_PIXEL_VALUE]))
        )

        # Calculate FFT magnitudes
        fft_magnitudes = np.abs(np.fft.fft(self.profile_resampled[self.COL_HANNING]))

        # Calculate FFT frequencies
        num_samples = len(fft_magnitudes)
        fft_frequencies = np.fft.fftfreq(n=num_samples, d=self.resampling_interval_mm)

        # Create a dataframe with the FFT data. Only keep positive frequencies above approximately DC (0 lp/mm)
        self.fft_data = pd.DataFrame({
            self.COL_FFT_MAGS: fft_magnitudes[2:num_samples // 2],
            self.COL_FFT_FREQS: fft_frequencies[2:num_samples // 2],
        })

        # Create a monotonic (doesn't 'overshoot') cubic spline to interpolate between discrete FFT results
        # noinspection PyUnresolvedReferences
        spline = scipy.interpolate.InterpolatedUnivariateSpline(
            x=self.fft_data[self.COL_FFT_FREQS],
            y=self.fft_data[self.COL_FFT_MAGS],
            ext='zeros',
        )

        # Save interpolated data in dataframe
        start_per_mm = math.ceil(self.fft_data[self.COL_FFT_FREQS].min() * 100) / 100  # round to 0.01
        end_per_mm = math.ceil(self.fft_data[self.COL_FFT_FREQS].max() * 100) / 100  # round to 0.01
        xs = np.arange(start_per_mm, end_per_mm, 0.01)
        ys = spline(xs)
        self.fft_data_interpolated = pd.DataFrame({
            self.COL_FFT_FREQS: xs,
            self.COL_FFT_MAGS: ys,
        })

        # Note data type in dataframes
        self.fft_data[self.COL_DATA_TYPE] = 'Discrete'
        self.fft_data_interpolated[self.COL_DATA_TYPE] = 'Interpolated'

        # Find maximum amplitude and associated frequency
        max_magnitude_index = self.fft_data_interpolated[self.COL_FFT_MAGS].idxmax()
        max_magnitude = self.fft_data_interpolated.loc[max_magnitude_index, self.COL_FFT_MAGS]
        self.measured_line_pairs_per_mm = self.fft_data_interpolated.loc[max_magnitude_index, self.COL_FFT_FREQS]

        if (abs(self.measured_line_pairs_per_mm / self.nominal_line_pairs_per_mm - 1.0) >
                self.ALLOWED_FREQUENCY_DETECTION_ERROR):
            # Too much variation between measured and expected frequencies. MTF = 0
            return 0.0
        else:
            # Approximate MTF as ratio of measured and ideal FFT magnitudes
            mtf = max_magnitude / self.ideal_fft_max_magnitude

            # Sometimes mtf can be > 1 due to pixel sampling phase issues (or if the profile approaches a square wave
            # instead of a sine wave. Set to one in these cases.
            return mtf if mtf <= 1.0 else 1.0

    def get_interactive_plot_profile(self) -> go.Figure:
        """Plotly scatter plot with profile data"""

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.profile_raw[self.COL_DISTANCE_MM],
            y=self.profile_raw[self.COL_PIXEL_VALUE],
            name='Original',
            mode='markers',
        ))

        fig.add_trace(go.Scatter(
            x=self.profile_resampled[self.COL_DISTANCE_MM],
            y=self.profile_resampled[self.COL_PIXEL_VALUE],
            name='Resampled',
            mode='markers',
        ))

        fig.update_layout(
            title=f'Profile (nominal lp/mm = {self.nominal_line_pairs_per_mm:.2f})',
            xaxis_title=self.COL_DISTANCE_MM,
            yaxis_title=self.COL_PIXEL_VALUE,
        )

        return fig

    def get_interactive_plot_fft(self) -> go.Figure:
        """Plotly scatter plot with fft data"""

        # make sure the MTF data is calculated
        mtf = self.mtf

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.fft_data[self.COL_FFT_FREQS],
            y=self.fft_data[self.COL_FFT_MAGS],
            name='Discrete',
            mode='markers',
        ))

        fig.add_trace(go.Scatter(
            x=self.fft_data_interpolated[self.COL_FFT_FREQS],
            y=self.fft_data_interpolated[self.COL_FFT_MAGS],
            name='Interpolated',
            mode='lines',
        ))

        fig.update_layout(
            title=f'FFT (nominal lp/mm = {self.nominal_line_pairs_per_mm:.2f})',
            xaxis=dict(range=ecu.pylinac_utils.leeds.LeedsTORUpdated.DEFAULT_LP_PER_MM_GRAPH_RANGE),
            xaxis_title=self.COL_FFT_FREQS,
            yaxis_title=self.COL_FFT_MAGS,
        )

        return fig


class HighContrastRectangularROIStrip(RectangularROI):
    """Rectangular region of interest covering multiple high contrast bar patterns in image."""

    all_nominal_line_pairs_per_mm: pydantic.conlist(pydantic.PositiveFloat, min_length=1) = pydantic.Field(
        default=None,
        description='The line-pairs-per-mm values for each bar pattern in the strip in the order that it appears'
    )

    all_nominal_line_pairs_per_region: pydantic.conlist(pydantic.PositiveInt, min_length=1) = pydantic.Field(
        default=None,
        description='The number of expected line-pairs in each region of the strip'
    )

    region_fractional_divisions: pydantic.conlist(pydantic.NonNegativeFloat, min_length=1) = pydantic.Field(
        default=None,
        description='Divisions between strip regions expressed as a fraction of the total strip length'
    )

    expected_max_pixel_value: float = pydantic.Field(
        default=None,
        description='Expected max pixel value in the high contrast bar pattern as measured on the phantom'
    )

    expected_min_pixel_value: float = pydantic.Field(
        default=None,
        description='Expected min pixel value in the high contrast bar pattern as measured on the phantom'
    )

    measured_upper_pixel_quantile: float = pydantic.Field(
        default=None,
        description='Measured upper pixel quantile value (should be close to expected max value)'
    )

    measured_lower_pixel_quantile: float = pydantic.Field(
        default=None,
        description='Measured lower pixel quantile value (should be close to expected min value)'
    )

    profile_raw: pd.DataFrame = pydantic.Field(
        default=None,
        description='A dataframe containing the 2D profile of the bar patterns in this ROI strip.'
    )

    profile_resampled: pd.DataFrame = pydantic.Field(
        default=None,
        description='A dataframe containing the 2D profile of the bar patterns in this ROI strip interpolated and '
                    'resampled into regularly spaced x intervals.'
    )

    profile_ref_pixel_value: float = pydantic.Field(
        default=None,
        description='Pixel value at which strip profile should be well-defined; measure reference length at this level'
    )

    profile_ref_start_mm: float = pydantic.Field(
        default=None,
        description='Reference left edge of strip profile; measured at reference pixel value'
    )

    profile_ref_start_index_lbl: int = pydantic.Field(
        default=None,
        description='Reference left edge of strip profile; measured at reference pixel value'
    )

    profile_ref_end_mm: float = pydantic.Field(
        default=None,
        description='Reference right edge of strip profile; measured at reference pixel value'
    )

    profile_ref_end_index_lbl: int = pydantic.Field(
        default=None,
        description='Reference right edge of strip profile; measured at reference pixel value'
    )

    regions: typing.List[HighContrastRectangularROI] = pydantic.Field(
        default=[],
        description='The list of high contrast rectangular ROI derived from this strip.'
    )

    pixel_oversampling_ratio: float = pydantic.Field(
        default=20.0,
        description='When resampling raw profile data, calculate the new grid at intervals equal to the width of one '
                    'pixel in mm divided by this number.'
    )

    @classmethod
    def define_relative_to_phantom(
            cls,
            location_radius_rel_phantom: float,
            location_angle_deg_rel_phantom: float,
            width_rel_phantom: float,
            height_rel_phantom: float,
            rotation_deg_rel_phantom: float) -> 'HighContrastRectangularROIStrip':
        return HighContrastRectangularROIStrip(
            location_radius_rel_phantom=location_radius_rel_phantom,
            location_angle_deg_rel_phantom=location_angle_deg_rel_phantom,
            width_rel_phantom=width_rel_phantom,
            height_rel_phantom=height_rel_phantom,
            rotation_deg_rel_phantom=rotation_deg_rel_phantom,
        )

    def _calc_reference_pixel_value(self):
        """Pixel value at which strip profile should be well-defined. Measure reference length at this level."""
        self.profile_ref_pixel_value = (self.profile_resampled[self.COL_PIXEL_VALUE].quantile(0.80) +
                                        self.profile_resampled[self.COL_PIXEL_VALUE].quantile(0.02)) / 2.0

    def _find_reference_edges(self):
        # Create a df from values above the threshold (maintains original index labels)
        df_above = self.profile_resampled[self.profile_resampled[self.COL_PIXEL_VALUE] > self.profile_ref_pixel_value]

        # Beginning crossing index label is first entry
        self.profile_ref_start_index_lbl = df_above.index[0]

        # End crossing index is the last entry
        self.profile_ref_end_index_lbl = df_above.index[-1]

        # Convert to mm
        self.profile_ref_start_mm = self.profile_resampled[self.COL_DISTANCE_MM].loc[self.profile_ref_start_index_lbl]
        self.profile_ref_end_mm = self.profile_resampled[self.COL_DISTANCE_MM].loc[self.profile_ref_end_index_lbl]

    def place_roi_on_image(self, image_array: pydantic_numpy.typing.NpNDArray,
                           phantom_center_px: pylinac.core.geometry.Point,
                           phantom_angle_deg_rel_image: float, phantom_radius_px: float, pixel_size_mm: float):

        # call super class method
        super().place_roi_on_image(
            image_array=image_array,
            phantom_center_px=phantom_center_px,
            phantom_angle_deg_rel_image=phantom_angle_deg_rel_image,
            phantom_radius_px=phantom_radius_px,
            pixel_size_mm=pixel_size_mm,
        )

        # sample pixels
        self._collect_pixels_into_2d_profile()

        # process profile into regularly spaced x-values
        self._interpolate_and_resample_profile()

        # determine approximate min and max pixel values
        self.measured_lower_pixel_quantile = self.profile_resampled[self.COL_PIXEL_VALUE].quantile(0.02)
        self.measured_upper_pixel_quantile = self.profile_resampled[self.COL_PIXEL_VALUE].quantile(0.98)

        # get reference pixel value
        self._calc_reference_pixel_value()

        # determine reference profile edges
        self._find_reference_edges()

        # calculate actual strip length
        strip_length_actual_mm = self.profile_ref_end_mm - self.profile_ref_start_mm

        #
        # divide strip into individual regions
        #

        # iterate over each region
        for region_index in range(len(self.all_nominal_line_pairs_per_mm)):
            # estimate position of space between regions
            left_edge_mm = self.profile_ref_start_mm + self.region_fractional_divisions[
                region_index] * strip_length_actual_mm
            right_edge_mm = self.profile_ref_start_mm + (self.region_fractional_divisions[region_index + 1] *
                                                         strip_length_actual_mm)

            # convert to index labels
            left_edge_index_lbl = (self.profile_resampled[self.COL_DISTANCE_MM] - left_edge_mm).abs().idxmin()
            right_edge_index_lbl = (self.profile_resampled[self.COL_DISTANCE_MM] - right_edge_mm).abs().idxmin()

            # create a new dataframe with these edges
            df_sampled = self.profile_resampled.loc[left_edge_index_lbl:right_edge_index_lbl,
                         [self.COL_DISTANCE_MM, self.COL_PIXEL_VALUE, self.COL_DATA_TYPE]]

            # tighten edges to actual profile
            start_index_lbl, end_index_lbl = HighContrastRectangularROI.find_profile_beg_and_end(
                df=df_sampled, target_pixel_value=self.profile_ref_pixel_value)
            df_sampled = df_sampled.loc[start_index_lbl:end_index_lbl]

            # snip ROI raw data as well
            left_raw_index_lbl = (self.profile_raw[self.COL_DISTANCE_MM] - df_sampled.loc[start_index_lbl,
            self.COL_DISTANCE_MM]).abs().idxmin()
            right_raw_index_lbl = (self.profile_raw[self.COL_DISTANCE_MM] - df_sampled.loc[end_index_lbl,
            self.COL_DISTANCE_MM]).abs().idxmin()
            df_raw = self.profile_raw.loc[left_raw_index_lbl:right_raw_index_lbl,
                     [self.COL_DISTANCE_MM, self.COL_PIXEL_VALUE, self.COL_DATA_TYPE]]

            # calculate center point for ROI
            start_mm = df_sampled[self.COL_DISTANCE_MM].iloc[0]
            end_mm = df_sampled[self.COL_DISTANCE_MM].iloc[-1]
            mid_mm = (start_mm + end_mm) / 2.0
            strip_roi_length_mm = self.profile_raw[self.COL_DISTANCE_MM].iloc[-1]
            fractional_distance = mid_mm / strip_roi_length_mm
            start_pt = self.pt_btm_middle_px
            end_pt = self.pt_top_middle_px
            center_pt_px = ((end_pt - start_pt).as_point() * fractional_distance + start_pt).as_point()

            # calculate resampling interval
            resampling_interval_mm = (end_mm - start_mm) / len(df_sampled[self.COL_PIXEL_VALUE])

            self.regions.append(
                HighContrastRectangularROI.from_strip_data(
                    profile_resampled=df_sampled,
                    resampling_interval_mm=resampling_interval_mm,
                    profile_raw=df_raw,
                    image_array=self.image_array,
                    nominal_line_pairs=self.all_nominal_line_pairs_per_region[region_index],
                    nominal_line_pairs_per_mm=self.all_nominal_line_pairs_per_mm[region_index],
                    nominal_max=self.expected_max_pixel_value,
                    nominal_min=self.expected_min_pixel_value,
                    pixel_size_mm=self.pixel_size_mm,
                    width_px=self.width_px,
                    height_px=(end_mm - start_mm) / self.pixel_size_mm,
                    center_pt_px=center_pt_px,
                    rotation_rad_rel_image=self.rotation_rad_rel_image,
                )
            )

    def _collect_pixels_into_2d_profile(self):
        """Collapse ROI into 2D array where x values are each pixel distance from one edge of the strip."""

        # bottom (starting) edge of the ROI strip
        # noinspection PyUnresolvedReferences
        baseline_px = pylinac.core.geometry.Line(self.pt_btm_left_px, self.pt_btm_right_px)

        # get coordinates for pixels in ROI
        rr, cc = self.masked_pixel_coordinates
        num_pixels = rr.shape[0]

        # allocate arrays to hold data (using default np.float64 datatype)
        values = np.empty(num_pixels)
        distances = np.empty(num_pixels)

        # calculate data for each pixel in ROI
        for i, (row, col) in enumerate(zip(rr, cc)):
            values[i] = self.image_array[row, col]
            # noinspection PyUnresolvedReferences
            distances[i] = baseline_px.distance_to(pylinac.core.geometry.Point(col, row))

        # convert to dataframe
        self.profile_raw = pd.DataFrame(data={self.COL_PIXEL_VALUE: values, self.COL_DISTANCE_PX: distances})
        self.profile_raw.sort_values(by=self.COL_DISTANCE_PX, inplace=True)

        # create a new column with distances in mm
        self.profile_raw[self.COL_DISTANCE_MM] = self.profile_raw[self.COL_DISTANCE_PX] * self.pixel_size_mm

        # label this profile as "raw"
        self.profile_raw[self.COL_DATA_TYPE] = 'Raw'

    def _interpolate_and_resample_profile(self):
        """Takes raw profile and samples the data onto a regularly spaced grid of x-values"""

        # Determine sampling interval
        sampling_interval_mm = self.pixel_size_mm / self.pixel_oversampling_ratio

        # Calculate x data
        start_mm = self.profile_raw[self.COL_DISTANCE_MM].iloc[0]
        end_mm = self.profile_raw[self.COL_DISTANCE_MM].iloc[-1]
        xs = np.linspace(
            start=start_mm,
            stop=end_mm,
            num=int((end_mm - start_mm) / sampling_interval_mm),
        )

        # Calculate y data
        ys = np.interp(
            x=xs,
            xp=self.profile_raw[self.COL_DISTANCE_MM],
            fp=self.profile_raw[self.COL_PIXEL_VALUE],
        )

        # Create dataframe for resampled profile
        self.profile_resampled = pd.DataFrame({
            self.COL_DISTANCE_MM: xs,
            self.COL_PIXEL_VALUE: ys,
            self.COL_DATA_TYPE: 'Resampled',
        })

    def plot_interactive_profile(self, show: bool = False) -> go.Figure:
        """Plotly graph of the profile"""

        # create a new dataframe from the raw and binned data
        combined_df = pd.concat([self.profile_raw, self.profile_resampled])

        fig = px.scatter(
            combined_df,
            x=self.COL_DISTANCE_MM,
            y=self.COL_PIXEL_VALUE,
            color=self.COL_DATA_TYPE,
        )

        if self.expected_max_pixel_value is not None:
            fig.add_hline(
                y=self.expected_max_pixel_value,
                line_dash='dot',
                annotation_text=f'expected max',
                annotation_position='top right',
            )

        if self.expected_min_pixel_value is not None:
            fig.add_hline(
                y=self.expected_min_pixel_value,
                line_dash='dot',
                annotation_text=f'expected min',
                annotation_position='top right',
            )

        if self.measured_lower_pixel_quantile is not None:
            fig.add_hline(
                y=self.measured_lower_pixel_quantile,
                line_dash='dash',
                line_color='blue',
                annotation_text=f'2%',
                annotation_position='top left',
                annotation_font_color='blue',
            )

        if self.measured_upper_pixel_quantile is not None:
            fig.add_hline(
                y=self.measured_upper_pixel_quantile,
                line_dash='dash',
                line_color='blue',
                annotation_text=f'98%',
                annotation_position='top left',
                annotation_font_color='blue',
            )

        if self.profile_ref_start_mm is not None and self.profile_ref_end_mm is not None:
            fig.add_shape(
                type="line",
                x0=self.profile_ref_start_mm,
                y0=self.profile_ref_pixel_value,
                x1=self.profile_ref_end_mm,
                y1=self.profile_ref_pixel_value,
                line=dict(
                    # color="DarkSlateGray",
                    # width=1,
                    dash="dot",
                )
            )

        for i in range(-1, len(self.regions)):
            left = 0 if i < 0 else self.regions[i].profile_resampled[self.COL_DISTANCE_MM].iloc[-1]
            right = self.regions[i + 1].profile_resampled[self.COL_DISTANCE_MM].iloc[0] if i < len(
                self.regions) - 1 else (
                self.profile_resampled[self.COL_DISTANCE_MM].iloc[-1])

            fig.add_vrect(
                x0=left,
                x1=right,
                fillcolor='black',
                opacity=0.2,
                line_width=0,
            )

        if show:
            fig.show()

        return fig
