import re
from typing import Optional

import numpy as np

from matplotlib.path import Path
from xml.etree.ElementTree import Element


def parse_polygon_points(polygon_xml: str) -> Optional[np.ndarray]:
    """
    Parse polygon points from XML string.
    
    Args:
        polygon_xml: XML string containing polygon points in format
                     'points="x1,y1;x2,y2;x3,y3;..."'
    
    Returns:
        Array of shape (N, 2) with (x, y) points, or None if parsing fails
    """
    # Extract points attribute from XML string using regex
    # Pattern matches: points="..." or points='...'
    match = re.search(r'points=["\']([^"\']+)["\']', polygon_xml)
    if not match:
        return None
    
    points_str = match.group(1)
    
    # Parse points: format is "x1,y1;x2,y2;x3,y3;..."
    # Replace semicolons with commas for easier parsing
    points_str = points_str.replace(';', ',')
    # Split by comma and convert to float array
    coords = [float(pt) for pt in points_str.split(',')]
    # Reshape to (N, 2) array of (x, y) points
    polygon_points = np.array(coords).reshape((-1, 2))
    
    return polygon_points


def intersects_polygon(left: float, top: float, right: float, bottom: float, polygon_xml: str) -> bool:
    """
    Check if a bounding box intersects with a polygon defined in XML format.
    
    Args:
        left: Left coordinate of bounding box
        top: Top coordinate of bounding box
        right: Right coordinate of bounding box
        bottom: Bottom coordinate of bounding box
        polygon_xml: XML string containing polygon points in format
                     'points="x1,y1;x2,y2;x3,y3;..."'
    
    Returns:
        True if the bounding box intersects with the polygon, False otherwise
    """
    # Parse polygon points from XML
    polygon_points = parse_polygon_points(polygon_xml)
    if polygon_points is None:
        return False
    
    # Create matplotlib Path from polygon points
    polygon_path = Path(polygon_points)
    
    # Get bounding box corners
    bbox_corners = np.array([
        [left, top],      # Top-left
        [right, top],     # Top-right
        [right, bottom],  # Bottom-right
        [left, bottom],   # Bottom-left
    ])
    
    # Check if any corner of the bounding box is inside the polygon
    if polygon_path.contains_points(bbox_corners).any():
        return True
    
    # Check if any vertex of the polygon is inside the bounding box
    # A point is inside a bounding box if: left <= x <= right and top <= y <= bottom
    polygon_inside_bbox = (
        (polygon_points[:, 0] >= left) & (polygon_points[:, 0] <= right) &
        (polygon_points[:, 1] >= top) & (polygon_points[:, 1] <= bottom)
    )
    if polygon_inside_bbox.any():
        return True
    
    return False


def get_mask(mask: Element, width: int, height: int):
    domain = mask.find('.//polygon[@label="domain"]')
    assert isinstance(domain, Element)
    domain = domain.attrib['points']
    domain = domain.replace(';', ',')
    domain = np.array([
        float(pt) for pt in domain.split(',')]).reshape((-1, 2))
    tl = (int(np.min(domain[:, 1])), int(np.min(domain[:, 0])))
    br = (int(np.max(domain[:, 1])), int(np.max(domain[:, 0])))
    domain_poly = Path(domain)
    # width, height = int(frame.shape[1]), int(frame.shape[0])
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T
    bitmap = domain_poly.contains_points(pixel_points)
    bitmap = bitmap.reshape((height, width, 1))
    # bitmap = bitmap[tl[0]:br[0], tl[1]:br[1], :]
    return bitmap, tl, br
