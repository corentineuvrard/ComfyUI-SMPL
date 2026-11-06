"""Views, cameras and keypoints of 'Render SMPL Views' / 'SMPL Pose Keypoints' (no model files needed)."""

import math
import types

import numpy as np
import pytest

from helpers import box_mesh
from smpl_avatar import keypoints, render, views
from smpl_avatar.render import OrthoView, PerspectiveView, RendererCache, render_mesh, view_from_array


def body_like_points():
    """A crude standing body: 1.7 m tall, 1.6 m arm span, 0.3 m deep."""
    rng = np.random.default_rng(0)
    points = rng.uniform([-0.8, 0.0, -0.15], [0.8, 1.7, 0.15], size=(400, 3))
    points[0], points[1] = [-0.8, 1.4, 0.0], [0.8, 1.4, 0.0]  # hands at the ends of the arms
    return points


# ------------------------------------------------------------------ presets


def test_presets_list_the_expected_views():
    assert [s.azimuth for s in views.expand("turnaround 4")] == [0.0, 90.0, 180.0, -90.0]
    assert [s.label for s in views.expand("turnaround 8")][:3] == ["front", "front 3/4 right", "right profile"]
    assert [s.azimuth for s in views.expand("turnaround 8")] == [0, 45, 90, 135, 180, -135, -90, -45]
    assert [s.label for s in views.expand("front / side / back")] == ["front", "right profile", "back"]
    assert views.expand("front 3/4 left")[0].azimuth == -45.0
    assert len(views.VIEW_PRESETS) == 10


def test_azimuth_turns_every_view_and_elevation_applies_to_all():
    specs = views.expand("turnaround 4", azimuth=30.0, elevation=15.0)
    assert [s.azimuth for s in specs] == [30.0, 120.0, -150.0, -60.0]
    assert {s.elevation for s in specs} == {15.0}
    assert specs[0].label == "front +30"  # a turned preset view says so
    assert views.expand("turnaround 4", elevation=15.0)[0].label == "front"
    (custom,) = views.expand("custom", azimuth=-120.0, elevation=-10.0)
    assert (custom.label, custom.azimuth, custom.elevation) == ("custom", -120.0, -10.0)


@pytest.mark.parametrize("kwargs, message", [
    ({"preset": "top"}, "unknown views"),
    ({"preset": "front", "azimuth": float("nan")}, "azimuth"),
    ({"preset": "front", "elevation": 90.0}, "elevation"),
])
def test_invalid_views_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        views.expand(**kwargs)


def test_view_rotations_follow_the_documented_conventions():
    pivot = np.zeros(3)
    right_side, front, top = np.array([[-1.0, 0, 0]]), np.array([[0, 0, 1.0]]), np.array([[0, 1.0, 0]])
    profile = views.ViewSpec("right profile", 90.0, 0.0)
    # The body's right side (-x, SMPL's left is +x) turns towards the camera (+z).
    np.testing.assert_allclose(views.transform(right_side, profile, pivot), [[0, 0, 1.0]], atol=1e-12)
    back = views.ViewSpec("back", 180.0, 0.0)
    np.testing.assert_allclose(views.transform(front, back, pivot), [[0, 0, -1.0]], atol=1e-12)
    above = views.ViewSpec("custom", 0.0, 30.0)
    moved = views.transform(top, above, pivot)[0]
    assert moved[2] > 0.49 and moved[1] == pytest.approx(math.cos(math.radians(30)))  # top tilts to the camera


# ------------------------------------------------------------------ framing


@pytest.mark.parametrize("projection", views.PROJECTIONS)
def test_fit_framing_keeps_every_view_inside_at_one_scale(projection):
    body = body_like_points()
    specs = views.expand("turnaround 8", elevation=10.0)
    pivot = views.pivot_of(body)
    posed = [views.transform(body, s, pivot) for s in specs]
    cameras = views.frame_views(posed, 384, 576, projection, 40.0, "fit", 0.1)
    for points, camera in zip(posed, cameras):
        assert views.outside_fraction(points, camera, 384, 576) == 0.0
        assert camera.znear < camera.camera_z - points[:, 2].max()
        assert camera.zfar > camera.camera_z - points[:, 2].min()
    # One scale and one vertical centre for the whole set (a character sheet).
    assert len({round(c.center_y, 9) for c in cameras}) == 1
    if projection == "orthographic":
        assert len({round(c.ymag, 9) for c in cameras}) == 1
    else:
        assert len({round(c.camera_z, 9) for c in cameras}) == 1


def test_fit_framing_is_tight_for_the_limiting_view():
    body = body_like_points()
    posed = [body]  # the front view alone: width-limited at 384 x 576
    (camera,) = views.frame_views(posed, 384, 576, "orthographic", framing="fit", padding=0.1)
    uv = camera.project(body, 384, 576)
    assert (uv[:, 0].max() - uv[:, 0].min()) / 384 == pytest.approx(1 / 1.1)


@pytest.mark.parametrize("projection", views.PROJECTIONS)
def test_fit_framing_without_padding_puts_the_body_on_the_border_not_outside(projection):
    # The outermost points land on the border, give or take a rounding error,
    # which must not count as a cut body.
    rng = np.random.default_rng(1)
    for width, height in ((384, 576), (512, 768), (1000, 336), (64, 4096)):
        for _ in range(25):
            body = rng.normal(size=(200, 3)) * rng.uniform(0.2, 1.0, size=3)
            specs = views.expand("turnaround 8", elevation=float(rng.uniform(-60.0, 60.0)))
            pivot = views.pivot_of(body)
            posed = [views.transform(body, s, pivot) for s in specs]
            cameras = views.frame_views(posed, width, height, projection, 40.0, "fit", 0.0)
            for points, camera in zip(posed, cameras):
                assert views.outside_fraction(points, camera, width, height) == 0.0


def test_fixed_framing_keeps_metres_per_pixel_and_the_feet_line():
    small = body_like_points() * [1, 0.9, 1]
    tall = body_like_points()
    a, = views.frame_views([small], 400, 800, "orthographic", framing="fixed", padding=0.1, view_height_m=2.2)
    b, = views.frame_views([tall], 400, 800, "orthographic", framing="fixed", padding=0.1, view_height_m=2.2)
    assert a.ymag == b.ymag == pytest.approx(1.1)
    feet = lambda cam, pts: cam.project(pts[pts[:, 1].argmin()][None], 400, 800)[0, 1]  # noqa: E731
    assert feet(a, small) == pytest.approx(feet(b, tall)) == pytest.approx(800 * (1 - 0.1 / 2))
    persp, = views.frame_views([tall], 400, 800, "perspective", 30.0, "fixed", 0.1, 2.2)
    body_z = (tall[:, 2].min() + tall[:, 2].max()) / 2.0
    half_height_at_body = (persp.camera_z - body_z) * math.tan(math.radians(15.0))
    assert half_height_at_body == pytest.approx(1.1)


def test_perspective_fixed_framing_puts_the_lowest_point_on_the_padding_line():
    body = body_like_points()
    pivot = views.pivot_of(body)
    for elevation in (0.0, -20.0, -35.0):
        specs = views.expand("turnaround 8", elevation=elevation)
        posed = [views.transform(body, s, pivot) for s in specs]
        cameras = views.frame_views(posed, 256, 384, "perspective", 60.0, "fixed", 0.1, 2.6)
        lowest = max(float(c.project(p, 256, 384)[:, 1].max()) for p, c in zip(posed, cameras))
        assert lowest == pytest.approx(384 * (1 - 0.1 / 2), abs=1e-6)


def test_perspective_fixed_framing_keeps_the_camera_in_front_of_the_body():
    body = body_like_points()
    pivot = views.pivot_of(body)
    posed = [views.transform(body, s, pivot) for s in views.expand("turnaround 4")]
    cameras = views.frame_views(posed, 64, 64, "perspective", 60.0, "fixed", 0.1, 0.5)
    z_front = max(p[:, 2].max() for p in posed)
    assert all(c.camera_z >= z_front + 0.05 - 1e-12 for c in cameras)
    assert all(np.isfinite(c.project(p, 64, 64)).all() for p, c in zip(posed, cameras))


def test_points_behind_a_perspective_camera_have_no_pixel():
    camera = PerspectiveView(0.0, 0.0, camera_z=1.0, yfov=1.0, aspect=1.0, znear=0.1, zfar=5.0)
    uv = camera.project(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]), 32, 32)
    assert np.isfinite(uv[0]).all() and np.isnan(uv[1]).all()
    assert views.outside_fraction(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]), camera, 32, 32) == 0.5


def test_fov_only_matters_for_perspective():
    (camera,) = views.frame_views([body_like_points()], 64, 64, "orthographic", fov_deg=float("nan"))
    assert isinstance(camera, OrthoView)


@pytest.mark.parametrize("kwargs, message", [
    ({"projection": "fisheye"}, "projection"),
    ({"framing": "zoom"}, "framing"),
    ({"projection": "perspective", "fov_deg": 0.0}, "fov"),
    ({"padding": -0.1}, "padding"),
    ({"view_height_m": float("inf")}, "view_height_m"),
])
def test_invalid_framing_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        views.frame_views([body_like_points()], 64, 64, **kwargs)


# ------------------------------------------------------------------ cameras


def test_perspective_projection_is_a_pinhole():
    camera = PerspectiveView(center_x=0.1, center_y=1.0, camera_z=5.0, yfov=math.radians(60), aspect=1.5,
                             znear=0.5, zfar=10.0)
    focal = 300 / 2 / math.tan(math.radians(30))
    uv = camera.project(np.array([[0.1, 1.0, 0.0], [1.1, 2.0, 1.0]]), 450, 300)
    np.testing.assert_allclose(uv[0], [225.0, 150.0])
    np.testing.assert_allclose(uv[1], [225.0 + focal / 4.0, 150.0 - focal / 4.0])
    raw = np.array([[0.0, 3.25]])
    np.testing.assert_array_equal(camera.distance_from_raw_depth(raw), [[0.0, 3.25]])


@pytest.mark.parametrize("camera", [
    OrthoView(0.2, 1.1, 0.6, 0.9, camera_z=2.0, znear=0.5, zfar=3.5),
    PerspectiveView(0.2, 1.1, camera_z=4.0, yfov=0.7, aspect=0.75, znear=0.3, zfar=6.0),
])
def test_views_survive_the_worker_serialisation(camera):
    assert view_from_array(camera.projection, camera.to_array()) == camera
    with pytest.raises(ValueError, match="projection"):
        view_from_array("fisheye", camera.to_array())


@pytest.mark.parametrize("projection", views.PROJECTIONS)
def test_described_camera_reproduces_the_projection(projection):
    """The JSON camera (OpenCV convention) maps a body point to the same pixel as the renderer."""
    body = body_like_points()
    spec = views.ViewSpec("custom", 35.0, 12.0)
    pivot = views.pivot_of(body)
    posed = views.transform(body, spec, pivot)
    (camera,) = views.frame_views([posed], 300, 420, projection, 45.0, "fit", 0.1)
    expected = camera.project(posed, 300, 420)
    described = views.describe(camera, spec, pivot, 300, 420)
    w2c = np.asarray(described["world_to_camera"])
    cam = (np.c_[body, np.ones(len(body))] @ w2c.T)[:, :3]
    assert (cam[:, 2] > 0).all()  # in front of the camera (z forward)
    if projection == "perspective":
        k = np.asarray(described["intrinsics"])
        uv = (cam / cam[:, 2:3]) @ k.T
        got = uv[:, :2]
    else:
        scale = described["pixels_per_metre"]
        got = np.asarray(described["principal_point"]) + scale * cam[:, :2]
    np.testing.assert_allclose(got, expected, atol=1e-3)


# ------------------------------------------------------------------ rendering


@pytest.fixture()
def cache():
    cache = RendererCache()
    yield cache
    cache.release()


def test_perspective_render_has_the_pinhole_size_and_true_distances(gl_available, cache):
    """A 1 m square 4 m away, 60 degree vertical field of view on 300 px."""
    vertices, faces = box_mesh(size=(1.0, 1.0, 0.2), center=(0.0, 0.0, -0.1))  # front face at z = 0
    camera = PerspectiveView(0.0, 0.0, camera_z=4.0, yfov=math.radians(60), aspect=1.0, znear=1.0, zfar=8.0)
    result = render_mesh(vertices, faces, 300, 300, camera, cache=cache, mode="inprocess")
    expected_side = 1.0 / 4.0 * camera.focal_px(300)
    assert result.coverage.sum() == pytest.approx(expected_side ** 2, rel=0.01)
    corners = camera.project(np.array([[-0.5, 0.5, 0.0], [0.5, -0.5, 0.0]]), 300, 300)
    cols = np.flatnonzero(result.mask.max(axis=0))
    assert abs(cols[0] - corners[0, 0]) < 1 and abs(cols[-1] + 1 - corners[1, 0]) < 1
    inner = result.coverage == 1.0
    np.testing.assert_allclose(result.distance[inner], 4.0, atol=2e-3)
    assert result.depth[inner].min() == pytest.approx(1.0)


def test_back_view_mirrors_the_front_view(gl_available, cache):
    """An off-centre block seen from the back appears on the other side of the image."""
    body, faces = box_mesh(size=(0.4, 1.6, 0.3))
    arm, arm_faces = box_mesh(size=(0.6, 0.1, 0.1), center=(0.5, 1.0 - 0.8, 0.0))
    vertices = np.concatenate([body, arm])
    faces = np.concatenate([faces, arm_faces + len(body)])
    pivot = views.pivot_of(vertices)
    specs = views.expand("front / side / back")
    posed = [views.transform(vertices, s, pivot) for s in specs]
    cameras = views.frame_views(posed, 96, 96, "orthographic", framing="fit", padding=0.05)
    masks = [render_mesh(v, faces, 96, 96, c, cache=cache, mode="inprocess").mask for v, c in zip(posed, cameras)]
    assert (masks[2] != masks[0][:, ::-1]).mean() < 0.005  # equal up to edge pixels at the threshold
    centroid = lambda m: (m.sum(axis=0) * np.arange(96)).sum() / m.sum()  # noqa: E731
    assert centroid(masks[0]) < 47.5 < centroid(masks[2])  # the big block left of the arm, then right
    assert not np.array_equal(masks[1], masks[0])


def test_worker_renders_perspective_views_like_the_main_process(gl_available, cache, monkeypatch):
    worker = render.RenderWorker()
    monkeypatch.setattr(render, "RENDER_WORKER", worker)
    try:
        vertices, faces = box_mesh(size=(1.0, 1.4, 0.6))
        vertices = render.rotate_y(vertices, 30.0)
        (camera,) = views.frame_views([vertices], 80, 96, "perspective", 50.0, "fit", 0.1)
        local = render_mesh(vertices, faces, 80, 96, camera, cache=cache, mode="inprocess")
        remote = render_mesh(vertices, faces, 80, 96, camera, mode="worker")
        assert isinstance(camera, PerspectiveView)
        np.testing.assert_allclose(remote.coverage, local.coverage, atol=1 / 255)
        np.testing.assert_allclose(remote.distance, local.distance, atol=1e-3)
    finally:
        worker.close()


# ------------------------------------------------------------------ keypoints


def test_openpose_tables_are_consistent():
    assert len(keypoints.COCO_NAMES) == len(keypoints.COLORS) == 18
    assert len(keypoints.LIMBS) == 17
    assert set(keypoints.COCO_FROM_JOINT) | set(keypoints.COCO_FROM_VERTEX) == set(range(18))
    assert keypoints.joint_names("SMPL")[12] == "neck" and len(keypoints.joint_names("SMPL")) == 24
    assert len(keypoints.joint_names("SMPL-H")) == 22


def test_skeleton_drawing_follows_the_openpose_colours():
    k = np.zeros((18, 3))
    k[1] = [50, 20, 1]  # neck
    k[2] = [30, 40, 1]  # right shoulder
    k[5] = [70, 40, 1]  # left shoulder, limb neck-left shoulder present
    image = keypoints.draw(k, 100, 80)
    assert image.shape == (80, 100, 3) and image.dtype == np.float32
    assert tuple(np.round(image[40, 30] * 255)) == keypoints.COLORS[2]  # dots in full colour
    middle = np.round(image[30, 60] * 255)  # on the neck-left shoulder limb
    assert tuple(middle) == tuple(int(c * 0.6) for c in keypoints.COLORS[1])
    assert image[75, 5].sum() == 0  # black background
    k[5, 2] = 0  # an absent keypoint is not drawn, nor are its limbs
    assert keypoints.draw(k, 100, 80)[30, 60].sum() == 0


def test_pose_keypoint_frame_layout():
    person = np.arange(54, dtype=float).reshape(18, 3)
    frame = keypoints.pose_keypoint_frame([person, person], 640, 480)
    assert frame["canvas_width"] == 640 and frame["canvas_height"] == 480
    assert len(frame["people"]) == 2
    assert frame["people"][0]["pose_keypoints_2d"] == list(range(54))
    assert frame["people"][1]["face_keypoints_2d"] is None


def test_pose_frames_shift_each_view_of_a_sheet():
    first = np.zeros((18, 3))
    first[1] = [10.0, 20.0, 1.0]  # only the neck is seen
    second = first.copy()
    second[2] = [30.0, 40.0, 1.0]
    sheet = types.SimpleNamespace(width=64, height=96, layout="sheet")
    (frame,) = keypoints.pose_frames(sheet, [first, second])
    assert frame["canvas_width"] == 128 and frame["canvas_height"] == 96
    a, b = (np.asarray(p["pose_keypoints_2d"]).reshape(18, 3) for p in frame["people"])
    np.testing.assert_array_equal(a, first)
    assert b[1].tolist() == [74.0, 20.0, 1.0] and b[2].tolist() == [94.0, 40.0, 1.0]
    assert (b[[0, 3, 4]] == 0).all()  # absent keypoints stay (0, 0, 0)
    batch = types.SimpleNamespace(width=64, height=96, layout="batch")
    frames = keypoints.pose_frames(batch, [first, second])
    assert len(frames) == 2 and frames[1]["canvas_width"] == 64
    np.testing.assert_array_equal(np.asarray(frames[1]["people"][0]["pose_keypoints_2d"]).reshape(18, 3), second)
