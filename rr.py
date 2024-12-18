from functools import lru_cache
from math import sqrt
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from supervision.annotators.base import BaseAnnotator, ImageType
from supervision.annotators.utils import (
    ColorLookup,
    Trace,
    resolve_color,
    resolve_text_background_xyxy,
)
from supervision.config import CLASS_NAME_DATA_FIELD, ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.detection.utils import (
    clip_boxes,
    mask_to_polygons,
    polygon_to_mask,
    spread_out_boxes
)
from supervision.draw.color import Color, ColorPalette
from supervision.draw.utils import draw_polygon, draw_rounded_rectangle, draw_text
from supervision.geometry.core import Point, Position, Rect
from supervision.utils.conversion import (
    ensure_cv2_image_for_annotation,
    ensure_pil_image_for_annotation,
)
from supervision.utils.image import (
    crop_image,
    letterbox_image,
    overlay_image,
    scale_image,
)
from supervision.utils.internal import deprecated
class RoundBoxAnnotator(BaseAnnotator):
    """
    A class for drawing filled bounding boxes with round edges on an image
    using provided detections. The boxes have solid edges and transparent fill.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        roundness: float = 0.6,
        fill_opacity: float = 0.3,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the bounding box lines.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
            roundness (float): Percent of roundness for edges of bounding box.
                Value must be float 0 < roundness <= 1.0
                By default roundness percent is calculated based on smaller side
                length (width or height).
            fill_opacity (float): Opacity of the fill color (0.0 to 1.0).
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup
        if not 0 < roundness <= 1.0:
            raise ValueError("roundness attribute must be float between (0, 1.0]")
        self.roundness: float = roundness
        if not 0 <= fill_opacity <= 1.0:
            raise ValueError("fill_opacity must be between 0.0 and 1.0")
        self.fill_opacity: float = fill_opacity

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with filled rounded bounding boxes
        based on the provided detections.

        Args:
            scene (ImageType): The image where rounded bounding boxes will be drawn.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.

        Returns:
            The annotated image
        """
        assert isinstance(scene, np.ndarray)
        result = scene.copy()
        
        for detection_idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )

            radius = (
                int((x2 - x1) // 2 * self.roundness)
                if abs(x1 - x2) < abs(y1 - y2)
                else int((y2 - y1) // 2 * self.roundness)
            )

            circle_coordinates = [
                ((x1 + radius), (y1 + radius)),
                ((x2 - radius), (y1 + radius)),
                ((x2 - radius), (y2 - radius)),
                ((x1 + radius), (y2 - radius)),
            ]

            # Create mask for the filled shape
            mask = np.zeros(scene.shape[:2], dtype=np.uint8)
            
            # Draw filled rectangles on mask
            cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 255, -1)
            cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 255, -1)
            
            # Draw filled circles at corners on mask
            for center in circle_coordinates:
                cv2.circle(mask, center, radius, 255, -1)

            # Create the fill color with transparency
            bgr_color = color.as_bgr()
            fill_color = (*bgr_color, int(255 * self.fill_opacity))  # BGRA
            
            # Only blend the filled area
            mask_area = mask == 255
            result_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
            result_rgba[mask_area] = cv2.addWeighted(
                result_rgba[mask_area],
                1 - self.fill_opacity,
                np.full_like(result_rgba[mask_area], fill_color),
                self.fill_opacity,
                0
            )
            result = cv2.cvtColor(result_rgba, cv2.COLOR_BGRA2BGR)

            # Draw the solid edges
            start_angles = (180, 270, 0, 90)
            end_angles = (270, 360, 90, 180)
            
            line_coordinates = [
                ((x1 + radius, y1), (x2 - radius, y1)),
                ((x2, y1 + radius), (x2, y2 - radius)),
                ((x1 + radius, y2), (x2 - radius, y2)),
                ((x1, y1 + radius), (x1, y2 - radius)),
            ]

            # Draw edges
            for center_coordinates, line, start_angle, end_angle in zip(
                circle_coordinates, line_coordinates, start_angles, end_angles
            ):
                cv2.ellipse(
                    img=result,
                    center=center_coordinates,
                    axes=(radius, radius),
                    angle=0,
                    startAngle=start_angle,
                    endAngle=end_angle,
                    color=bgr_color,
                    thickness=self.thickness,
                )

                cv2.line(
                    img=result,
                    pt1=line[0],
                    pt2=line[1],
                    color=bgr_color,
                    thickness=self.thickness,
                )

        return result
        
bounding_box_annotator = RoundBoxAnnotator(fill_opacity=0.3, thickness=2, roundness=0.2)
annotated_frame = bounding_box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

sv.plot_image(annotated_frame)
