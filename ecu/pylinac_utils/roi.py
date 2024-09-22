import math

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import scipy.interpolate

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

    # Constants for labeling Pandas columns
    COL_PIXEL_VALUE: typing.ClassVar[str] = 'Pixel Value'
    COL_DISTANCE_PX: typing.ClassVar[str] = 'Distance (Pixels)'
    COL_DISTANCE_MM: typing.ClassVar[str] = 'Distance (mm)'
    COL_DATA_TYPE: typing.ClassVar[str] = 'Data'

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
            line_color: str = 'White',
            opacity: float = 1.0,
            name: str = 'Rectangular ROI',
            show_legend: bool = True,
            legend_group_title: str = None,
            show_arrow: bool = True,
    ) -> None:
        """Add this ROI to the interactive plot."""

        arrow_delta = self.width_px if self.width_px < self.height_px else self.height_px
        arrow_delta *= 0.25
        arrow_angle_rad = np.deg2rad(45.0)
        # noinspection PyUnresolvedReferences
        arrow_pt1 = pylinac.core.geometry.Point(
            self.center_pt_px.x + arrow_delta * math.sin(self.rotation_rad_rel_image + arrow_angle_rad),
            self.center_pt_px.y + arrow_delta * math.cos(self.rotation_rad_rel_image + arrow_angle_rad),
        )
        # noinspection PyUnresolvedReferences
        arrow_pt2 = pylinac.core.geometry.Point(
            self.center_pt_px.x + arrow_delta * math.sin(self.rotation_rad_rel_image - arrow_angle_rad),
            self.center_pt_px.y + arrow_delta * math.cos(self.rotation_rad_rel_image - arrow_angle_rad),
        )

        svg_path = ''
        svg_path += (f'M {self.pt_btm_left_px.x}, {self.pt_btm_left_px.y} ' +
                     f'L {self.pt_top_left_px.x}, {self.pt_top_left_px.y} ' +
                     f'L {self.pt_top_right_px.x}, {self.pt_top_right_px.y} ' +
                     f'L {self.pt_btm_right_px.x}, {self.pt_btm_right_px.y} ')

        if show_arrow:
            svg_path += (
                    f'L {self.pt_btm_left_px.x}, {self.pt_btm_left_px.y} ' +
                    f'M {self.pt_btm_middle_px.x}, {self.pt_btm_middle_px.y} ' +
                    f'L {self.center_pt_px.x}, {self.center_pt_px.y} ' +
                    f'L {arrow_pt1.x}, {arrow_pt1.y} ' +
                    f'M {self.center_pt_px.x}, {self.center_pt_px.y} ' +
                    f'L {arrow_pt2.x}, {arrow_pt2.y} ')
        else:
            svg_path += 'Z'

        fig.add_shape(
            type='path',
            path=svg_path,
            line_color=line_color,
            opacity=opacity,
            showlegend=show_legend,
            name=name,
            legendgroup=legend_group_title,
            legendgrouptitle_text=legend_group_title,
        )

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
    nominal_line_pairs_per_mm: float = pydantic.Field(frozen=True)
    profile_resampled: pd.DataFrame = pydantic.Field(frozen=True)
    profile_raw: pd.DataFrame = pydantic.Field(frozen=True)

    @classmethod
    def from_strip_data(cls,
                        nominal_line_pairs_per_mm: float,
                        center_pt_px: pylinac.core.geometry.Point,
                        width_px: float,
                        height_px: float,
                        image_array: np.ndarray,
                        pixel_size_mm: float,
                        rotation_rad_rel_image: float,
                        profile_resampled: pd.DataFrame,
                        profile_raw: pd.DataFrame,
                        ) -> 'HighContrastRectangularROI':
        return HighContrastRectangularROI(
            nominal_line_pairs_per_mm=nominal_line_pairs_per_mm,
            center_pt_px=center_pt_px,
            width_px=width_px,
            height_px=height_px,
            image_array=image_array,
            pixel_size_mm=pixel_size_mm,
            rotation_rad_rel_image=rotation_rad_rel_image,
            profile_resampled=profile_resampled,
            profile_raw=profile_raw,
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
    def mtf(self) -> float:
        """Modulation Transfer Function for the ROI


        """

        # TODO
        return 1.0


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
            left_raw_index_lbl = (self.profile_raw[self.COL_DISTANCE_MM] - left_edge_mm).abs().idxmin()
            right_raw_index_lbl = (self.profile_raw[self.COL_DISTANCE_MM] - right_edge_mm).abs().idxmin()
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

            self.regions.append(
                HighContrastRectangularROI.from_strip_data(
                    profile_resampled=df_sampled,
                    profile_raw=df_raw,
                    image_array=self.image_array,
                    nominal_line_pairs_per_mm=self.all_nominal_line_pairs_per_mm[region_index],
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
