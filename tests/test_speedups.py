"""
check that performance optimizations give identical results
to the original (straightforward but slow) implementations
"""
import cv2
import numpy as np
import pytest

from eynollah.eynollah import normalize_image
from eynollah.utils import small_textlines_to_parent_adherence2
from eynollah.utils.rotate import rotate_image
from eynollah.utils.separate_lines import RotatedProjection, return_deskew_slop


def make_textline_mask(seed, height=300, width=200, angle=3.0):
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width), dtype=np.uint8)
    for y in range(15, height - 15, 18):
        x0 = rng.integers(5, 40)
        x1 = rng.integers(width - 40, width - 5)
        img[y: y + rng.integers(5, 10), x0: x1] = 1
    return rotate_image(img, angle)

def reference_profile(img, angle):
    # original implementation: rotate the full padded canvas
    height, width = img.shape[:2]
    max_shape = int(np.max(img.shape) * 1.1)
    onset_x = int(0.5 * (max_shape - width))
    onset_y = int(0.5 * (max_shape - height))
    img_resized = np.zeros((max_shape, max_shape))
    img_resized[onset_y: onset_y + height,
                onset_x: onset_x + width] = img
    img_rot = rotate_image(img_resized, angle)
    img_rot[img_rot != 0] = 1
    return img_rot.sum(axis=1)

@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("shape", [(300, 200), (120, 400), (500, 60)])
def test_rotated_projection_identical(seed, shape):
    img = make_textline_mask(seed, *shape, angle=seed * 2.5 - 2)
    projection = RotatedProjection(img)
    for angle in np.linspace(-90, 90, 41):
        assert np.array_equal(projection.profile(angle),
                              reference_profile(img, angle)), angle

def test_rotated_projection_empty():
    img = np.zeros((50, 80), dtype=np.uint8)
    assert not RotatedProjection(img).profile(10).any()
    assert return_deskew_slop(img, 2) == 0

@pytest.mark.parametrize("angle", [-4.0, 0.0, 1.5, 7.0])
def test_deskew_finds_angle(angle):
    img = make_textline_mask(0, 400, 400, angle=angle)
    # (slope is the rotation that undoes the skew)
    assert abs(return_deskew_slop(img, 2) + angle) < 0.5

def test_normalize_image_identical():
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    img = np.dstack([img, img[::-1], img.T])
    reference = (img / 255.).astype(np.float16)
    assert np.array_equal(normalize_image(img).view(np.uint16),
                          reference.view(np.uint16))
    assert normalize_image(img.astype(float)).dtype == np.float16

def reference_adherence(textlines_con, textline_mask, num_col):
    # original implementation: full-page masks for every pair of small/large contour
    textlines_con_new = []
    for region in textlines_con:
        areas = np.array(list(map(cv2.contourArea, region))) / float(textline_mask.size)
        min_area = 0.0004 if num_col == 0 else 0.0003 if num_col == 1 else 0.0001
        small = [c for c, a in zip(region, areas) if a < min_area]
        large = [c for c, a in zip(region, areas) if a >= min_area]
        img_small = cv2.fillPoly(np.zeros_like(textline_mask), pts=small, color=1)
        img_large = cv2.fillPoly(np.zeros_like(textline_mask), pts=large, color=1)
        if np.any(img_small + img_large == 2):
            inter = []
            for contour_small in small:
                intersections = []
                for contour_large in large:
                    img0_small = cv2.fillPoly(np.zeros_like(textline_mask), pts=[contour_small], color=1)
                    img0_large = cv2.fillPoly(np.zeros_like(textline_mask), pts=[contour_large], color=1)
                    intersections.append(np.count_nonzero(img0_small + img0_large == 2))
                idx = np.argmax(intersections)
                inter.append(idx if intersections[idx] > 0 else -1)
            inter = np.array(inter)
            for idx in set(inter):
                if idx < 0:
                    continue
                img0_union = cv2.fillPoly(np.zeros_like(textline_mask), pts=[large[idx]], color=255)
                for idx_small in np.flatnonzero(inter == idx):
                    img0_union = cv2.fillPoly(img0_union, pts=[small[idx_small]], color=255)
                contours, _ = cv2.findContours(img0_union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                large[idx] = contours[np.argmax(list(map(cv2.contourArea, contours)))]
        textlines_con_new.append(large)
    return textlines_con_new

@pytest.mark.parametrize("seed", range(5))
def test_small_textlines_adherence_identical(seed):
    rng = np.random.default_rng(seed)
    mask = np.zeros((400, 300), dtype=np.uint8)
    regions = []
    for _ in range(3):
        region = []
        for _ in range(12):
            # large lines and small fragments (some overlapping, some at the border)
            x, y = rng.integers(-5, 290), rng.integers(-5, 390)
            w, h = (rng.integers(40, 150), rng.integers(8, 15)) if rng.random() < 0.5 else \
                (rng.integers(3, 12), rng.integers(3, 12))
            region.append(np.array([[[x, y]], [[x + w, y + 1]], [[x + w, y + h]], [[x, y + h]]],
                                   dtype=np.int32))
        regions.append(region)
    result = small_textlines_to_parent_adherence2([list(r) for r in regions], mask, 3)
    expected = reference_adherence([list(r) for r in regions], mask, 3)
    assert len(result) == len(expected)
    for region_result, region_expected in zip(result, expected):
        assert len(region_result) == len(region_expected)
        for contour_result, contour_expected in zip(region_result, region_expected):
            assert np.array_equal(contour_result, contour_expected)
