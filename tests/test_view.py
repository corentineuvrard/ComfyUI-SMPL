import numpy as np
import pytest

from smpl_avatar.render import OrthoView, compute_view, rotate_y, vertex_normals
from helpers import box_mesh


def body_like_points():
    return np.array([[-0.9, -1.0, -0.1], [0.9, 0.8, 0.2]])


def test_fit_framing_contains_body_with_square_pixels():
    points = body_like_points()
    view = compute_view(points, 512, 768, framing="fit", padding=0.1)
    assert view.xmag / view.ymag == pytest.approx(512 / 768)
    pixels = view.project(points, 512, 768)
    assert (pixels[:, 0] >= 0).all() and (pixels[:, 0] <= 512).all()
    assert (pixels[:, 1] >= 0).all() and (pixels[:, 1] <= 768).all()
    # Width-limited here: the body spans 1 / 1.1 of the image width.
    assert (pixels[1, 0] - pixels[0, 0]) / 512 == pytest.approx(1 / 1.1)
    assert view.znear < view.camera_z - points[:, 2].max()
    assert view.zfar > view.camera_z - points[:, 2].min()


def test_fixed_framing_keeps_metres_per_pixel():
    small = np.array([[-0.2, 0.0, 0.0], [0.2, 1.5, 0.1]])
    tall = np.array([[-0.2, 0.0, 0.0], [0.2, 1.9, 0.1]])
    a = compute_view(small, 400, 800, framing="fixed", padding=0.1, view_height_m=2.2)
    b = compute_view(tall, 400, 800, framing="fixed", padding=0.1, view_height_m=2.2)
    assert a.ymag == b.ymag == pytest.approx(1.1)
    feet_a = a.project(small[:1], 400, 800)[0, 1]
    feet_b = b.project(tall[:1], 400, 800)[0, 1]
    assert feet_a == pytest.approx(feet_b) == pytest.approx(800 * (1 - 0.1 / 2))
    head_a = a.project(small[1:], 400, 800)[0, 1]
    head_b = b.project(tall[1:], 400, 800)[0, 1]
    assert head_a - head_b == pytest.approx(0.4 / 2.2 * 800)


def test_invalid_framing():
    with pytest.raises(ValueError):
        compute_view(body_like_points(), 10, 10, framing="zoom")
    with pytest.raises(ValueError):
        compute_view(body_like_points(), 0, 10)


def test_distance_inverts_pyrender_linearisation():
    view = OrthoView(0, 0, 1, 1, camera_z=3.0, znear=0.5, zfar=4.0)
    n, f = view.znear, view.zfar
    distance = np.array([[0.5, 1.2, 2.7, 4.0]])
    ndc = (2 * distance - f - n) / (f - n)  # orthographic projection
    pyrender_raw = 2 * n * f / (f + n - ndc * (f - n))  # what pyrender returns
    pyrender_raw = np.concatenate([pyrender_raw, [[0.0]]], axis=1)  # background
    np.testing.assert_allclose(view.distance_from_raw_depth(pyrender_raw), [[0.5, 1.2, 2.7, 4.0, 0.0]])


def test_rotate_y():
    point = np.array([[1.0, 2.0, 0.0]])
    np.testing.assert_allclose(rotate_y(point, 90), [[0.0, 2.0, -1.0]], atol=1e-12)
    np.testing.assert_array_equal(rotate_y(point, 0), point)


def test_vertex_normals_point_outwards():
    vertices, faces = box_mesh()
    normals = vertex_normals(vertices, faces)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
    assert (np.sum(normals * vertices, axis=1) > 0).all()
