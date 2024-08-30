import pylinac
import numpy as np
import dataclasses
import functools
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches


@dataclasses.dataclass(frozen=True)
class AnnulusROI():
    """A class representing an annulus-shaped Region of Interest."""

    image_array: np.ndarray
    angle_deg: float
    dist_from_center_px: float
    roi_outer_radius_px: float
    roi_inner_radius_px: float
    phantom_center_pt_px: pylinac.core.geometry.Point

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
