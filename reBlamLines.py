import bpy
import gpu
import math
import mathutils
from bpy_extras import view3d_utils
from bpy_extras.object_utils import world_to_camera_view
from gpu_extras.batch import batch_for_shader

bl_info = {
    'name': 'reBlam Lines - Perspective Matcher',
    'author': 'Per Gantelius / reBlam contributors',
    'version': (0, 3, 0),
    'blender': (4, 5, 0),
    'location': '3D View > Sidebar > reBlam Lines',
    'description': 'Camera calibration from custom drawn perspective guide lines.',
    'category': '3D View'
}

AXIS_COLORS = {
    'X': (1.0, 0.12, 0.08, 1.0),
    'Y': (0.1, 0.85, 0.18, 1.0),
    'Z': (0.18, 0.38, 1.0, 1.0),
}
FAMILY_TO_AXIS = ['X', 'Y', 'Z']
UNCLASSIFIED_COLOR = (0.78, 0.78, 0.78, 1.0)

_draw_handle = None
_timer_running = False
_line_editor_running = False


def length2(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1])


def normalize2(v):
    l = length2(v)
    if l < 1e-10:
        return (0.0, 0.0)
    return (v[0] / l, v[1] / l)


def dot2(a, b):
    return a[0] * b[0] + a[1] * b[1]


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def line_from_points(a, b):
    d = (b[0] - a[0], b[1] - a[1])
    if length2(d) < 1e-8:
        return None
    d = normalize2(d)
    n = (d[1], -d[0])
    c = dot2(n, a)
    return {'a': a, 'b': b, 'dir': d, 'normal': n, 'c': c}


def relative_to_image_plane(u, v, scene):
    aspect = scene.render.resolution_x / max(scene.render.resolution_y, 1)
    if aspect <= 1:
        return ((-1.0 + 2.0 * u) * aspect, 2.0 * v - 1.0)
    return (-1.0 + 2.0 * u, (2.0 * v - 1.0) / aspect)


def image_plane_to_relative(point, scene):
    aspect = scene.render.resolution_x / max(scene.render.resolution_y, 1)
    if aspect <= 1:
        return (0.5 * (point[0] / max(aspect, 1e-8) + 1.0), 0.5 * (point[1] + 1.0))
    return (0.5 * (point[0] + 1.0), 0.5 * (point[1] * aspect + 1.0))


def intersect_lines(l1, l2):
    n1, n2 = l1['normal'], l2['normal']
    det = cross2(n1, n2)
    if abs(det) < 1e-5:
        return None, abs(det)
    c1, c2 = l1['c'], l2['c']
    return ((c1 * n2[1] - n1[1] * c2) / det, (n1[0] * c2 - c1 * n2[0]) / det), abs(det)


def point_line_distance(point, line):
    return abs(dot2(line['normal'], point) - line['c'])


def segment_midpoint(line):
    return ((line['a'][0] + line['b'][0]) * 0.5, (line['a'][1] + line['b'][1]) * 0.5)


def vp_outside_score(vp):
    x, y = vp
    return max(abs(x) - 0.75, 0.0) + max(abs(y) - 0.55, 0.0)


def solve_vp_from_lines(lines):
    if len(lines) < 2:
        return None
    a00 = a01 = a11 = b0 = b1 = 0.0
    for line in lines:
        nx, ny = line['normal']
        c = line['c']
        a00 += nx * nx
        a01 += nx * ny
        a11 += ny * ny
        b0 += nx * c
        b1 += ny * c
    det = a00 * a11 - a01 * a01
    if abs(det) < 1e-9:
        return None
    return ((b0 * a11 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det)


def line_residual(vp, lines):
    if not vp or not lines:
        return 999.0
    return sum(point_line_distance(vp, line) for line in lines) / len(lines)


def camera_frame_points(scene, cam):
    frame = cam.data.view_frame(scene=scene)
    return [cam.matrix_world @ corner for corner in frame]


def uv_to_world(scene, cam, u, v):
    frame = camera_frame_points(scene, cam)
    top = frame[3].lerp(frame[0], u)
    bottom = frame[2].lerp(frame[1], u)
    return bottom.lerp(top, v)


def uv_to_region(scene, cam, region, region_data, u, v):
    world = uv_to_world(scene, cam, u, v)
    point = view3d_utils.location_3d_to_region_2d(region, region_data, world)
    if point is None:
        return None
    return (point.x, point.y)


def region_to_uv(scene, cam, region, region_data, x, y):
    origin = view3d_utils.region_2d_to_origin_3d(region, region_data, (x, y))
    direction = view3d_utils.region_2d_to_vector_3d(region, region_data, (x, y))
    frame = camera_frame_points(scene, cam)
    plane_point = frame[0]
    plane_normal = (cam.matrix_world.to_3x3() @ mathutils.Vector((0, 0, -1))).normalized()
    denom = direction.dot(plane_normal)
    if abs(denom) < 1e-8:
        return None
    hit = origin + direction * ((plane_point - origin).dot(plane_normal) / denom)
    local = cam.matrix_world.inverted() @ hit
    local_frame = cam.data.view_frame(scene=scene)
    xs = [corner.x for corner in local_frame]
    ys = [corner.y for corner in local_frame]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    u = (local.x - min_x) / max(max_x - min_x, 1e-8)
    v = (local.y - min_y) / max(max_y - min_y, 1e-8)
    return (clamp(u), clamp(v))


def draw_line_segments(segments, color, width):
    if not segments:
        return
    coords = []
    for a, b in segments:
        coords.extend([a, b])
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": coords})
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(width)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def draw_points(points, color, size):
    if not points:
        return
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'POINTS', {"pos": points})
    gpu.state.blend_set('ALPHA')
    gpu.state.point_size_set(size)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.point_size_set(1.0)
    gpu.state.blend_set('NONE')


def world_to_region(region, region_data, point):
    projected = view3d_utils.location_3d_to_region_2d(region, region_data, point)
    if projected is None:
        return None
    return (projected.x, projected.y)


def line_color(line):
    if line.axis in AXIS_COLORS:
        return AXIS_COLORS[line.axis]
    return UNCLASSIFIED_COLOR


def visible_plane_bounds(scene, cam):
    axis_a, axis_b = axis_names_from_preset(scene)
    ax_a = axis_index(axis_a)
    ax_b = axis_index(axis_b)
    normal_index = plane_normal_axis(axis_a, axis_b)
    plane_value = getattr(scene, "reblam_lines_plane_offset", 0.0)

    hits = []
    samples = 12
    for y in range(samples + 1):
        v = y / samples
        for x in range(samples + 1):
            u = x / samples
            if x not in {0, samples} and y not in {0, samples}:
                continue
            hit = camera_uv_ray_plane_intersection(scene, cam, u, v, normal_index, plane_value)
            if hit:
                hits.append(hit)

    if len(hits) >= 2:
        min_a = min(point[ax_a] for point in hits)
        max_a = max(point[ax_a] for point in hits)
        min_b = min(point[ax_b] for point in hits)
        max_b = max(point[ax_b] for point in hits)
    else:
        span = max(scene.reblam_lines_camera_distance, 1.0)
        min_a, max_a = -span, span
        min_b, max_b = -span, span

    extent = max(getattr(scene, "reblam_lines_plane_extent", 1.0), 0.05)
    span_a = max(max_a - min_a, 1.0)
    span_b = max(max_b - min_b, 1.0)
    pad_a = span_a * max(extent - 1.0, 0.0) * 0.5
    pad_b = span_b * max(extent - 1.0, 0.0) * 0.5
    min_a -= pad_a
    max_a += pad_a
    min_b -= pad_b
    max_b += pad_b
    return axis_a, axis_b, ax_a, ax_b, normal_index, plane_value, min_a, max_a, min_b, max_b, bool(hits)


def draw_plane_grid(scene, cam, region, region_data):
    if not getattr(scene, "reblam_lines_show_plane_grid", True):
        return
    axis_a, axis_b, ax_a, ax_b, normal_index, plane_value, min_a, max_a, min_b, max_b, _has_hits = visible_plane_bounds(scene, cam)
    divisions = 16
    segments_a = []
    segments_b = []

    def plane_point(value_a, value_b):
        coords = [0.0, 0.0, 0.0]
        coords[ax_a] = value_a
        coords[ax_b] = value_b
        coords[normal_index] = plane_value
        return mathutils.Vector(coords)

    for index in range(divisions + 1):
        t = index / divisions
        value_a = min_a + (max_a - min_a) * t
        value_b = min_b + (max_b - min_b) * t
        p1 = world_to_region(region, region_data, plane_point(value_a, min_b))
        p2 = world_to_region(region, region_data, plane_point(value_a, max_b))
        if p1 and p2:
            segments_a.append((p1, p2))
        p3 = world_to_region(region, region_data, plane_point(min_a, value_b))
        p4 = world_to_region(region, region_data, plane_point(max_a, value_b))
        if p3 and p4:
            segments_b.append((p3, p4))

    alpha = getattr(scene, "reblam_lines_grid_opacity", 0.45)
    width = getattr(scene, "reblam_lines_grid_width", 2.0)
    draw_line_segments(segments_a, (AXIS_COLORS[axis_a][0], AXIS_COLORS[axis_a][1], AXIS_COLORS[axis_a][2], alpha), width)
    draw_line_segments(segments_b, (AXIS_COLORS[axis_b][0], AXIS_COLORS[axis_b][1], AXIS_COLORS[axis_b][2], alpha), width)


def draw_guides_callback():
    context = bpy.context
    scene = context.scene
    cam = scene.camera
    if not cam or not getattr(scene, "reblam_lines_show_guides", True):
        return
    area = context.area
    region = context.region
    space = context.space_data
    if not area or area.type != 'VIEW_3D' or not region or not space or not space.region_3d:
        return

    active_index = scene.reblam_lines_active_index
    for index, guide in enumerate(scene.reblam_lines):
        p1 = uv_to_region(scene, cam, region, space.region_3d, guide.x1, guide.y1)
        p2 = uv_to_region(scene, cam, region, space.region_3d, guide.x2, guide.y2)
        if p1 is None or p2 is None:
            continue
        color = line_color(guide)
        width = scene.reblam_lines_width + (1.5 if index == active_index else 0.0)
        draw_line_segments([(p1, p2)], color, width)
        draw_points([p1, p2], color, max(width + 4.0, 7.0))
        if index == active_index:
            draw_points([p1, p2], (1.0, 1.0, 1.0, 0.95), 3.0)

    origin = uv_to_region(scene, cam, region, space.region_3d, scene.reblam_lines_origin_x, scene.reblam_lines_origin_y)
    if origin:
        ox, oy = origin
        size = max(scene.reblam_lines_width + 5.0, 9.0)
        draw_line_segments([((ox - size, oy), (ox + size, oy)), ((ox, oy - size), (ox, oy + size))], (1.0, 1.0, 1.0, 0.9), 2.0)
    draw_plane_grid(scene, cam, region, space.region_3d)


def tag_view3d_redraw():
    global _timer_running
    if _draw_handle is None:
        _timer_running = False
        return None
    wm = bpy.context.window_manager
    if not wm:
        return None
    for window in wm.windows:
        if not window.screen:
            continue
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    return 0.05 if getattr(bpy.context.scene, "reblam_lines_show_guides", True) else 0.25


def start_line_editor(context):
    global _line_editor_running
    if _line_editor_running:
        return
    if not context or not context.window_manager or not context.scene.camera:
        return
    if not context.area or context.area.type != 'VIEW_3D':
        return
    try:
        bpy.ops.reblam_lines.edit_lines('INVOKE_DEFAULT')
    except Exception:
        _line_editor_running = False


def ensure_draw_handler():
    global _draw_handle, _timer_running
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(draw_guides_callback, (), 'WINDOW', 'POST_PIXEL')
    if not _timer_running:
        bpy.app.timers.register(tag_view3d_redraw, persistent=True)
        _timer_running = True


def remove_draw_handler():
    global _draw_handle, _timer_running, _line_editor_running
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    _timer_running = False
    _line_editor_running = False


def collect_projected_lines(scene):
    items = []
    for index, guide in enumerate(scene.reblam_lines):
        a = relative_to_image_plane(guide.x1, guide.y1, scene)
        b = relative_to_image_plane(guide.x2, guide.y2, scene)
        line = line_from_points(a, b)
        if line:
            line['index'] = index
            line['guide'] = guide
            line['axis'] = guide.axis
            items.append(line)
    return items


def infer_vanishing_families(lines, threshold, max_families=3):
    candidates = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            vp, quality = intersect_lines(lines[i], lines[j])
            if vp is None:
                continue
            inliers = [line for line in lines if point_line_distance(vp, line) <= threshold]
            if len(inliers) < 2:
                continue
            refined = solve_vp_from_lines(inliers) or vp
            residual = line_residual(refined, inliers)
            alignment = abs(dot2(lines[i]['dir'], lines[j]['dir']))
            midpoint_distance = min(length2((refined[0] - segment_midpoint(line)[0], refined[1] - segment_midpoint(line)[1])) for line in inliers)
            score = (
                len(inliers) * 12.0
                + alignment * 8.0
                + min(vp_outside_score(refined), 4.0) * 1.5
                + min(midpoint_distance, 4.0) * 0.6
                - residual * 30.0
                - quality * 0.8
            )
            candidates.append({'vp': refined, 'inliers': inliers, 'score': score, 'residual': residual})

    candidates.sort(key=lambda c: c['score'], reverse=True)
    families = []
    used = set()
    for candidate in candidates:
        fresh = [line for line in candidate['inliers'] if line['index'] not in used]
        if len(fresh) < 2:
            continue
        refined = solve_vp_from_lines(fresh) or candidate['vp']
        families.append({
            'vp': refined,
            'lines': fresh,
            'residual': line_residual(refined, fresh),
        })
        used.update(line['index'] for line in fresh)
        if len(families) >= max_families:
            break
    return families


def classify_lines(scene, families, axis_order):
    for guide in scene.reblam_lines:
        if guide.axis == 'AUTO':
            guide.family = -1
    for family_index, family in enumerate(families):
        inferred_axis = family.get('axis')
        if not inferred_axis:
            inferred_axis = axis_order[family_index] if family_index < len(axis_order) else 'AUTO'
            family['axis'] = inferred_axis
        for line in family['lines']:
            guide = line['guide']
            guide.family = family_index
            if guide.axis == 'AUTO':
                guide.axis = inferred_axis


def axis_order_for_preset(scene):
    preset = scene.reblam_lines_axis_preset
    if preset == 'XY':
        return ['X', 'Y', 'Z']
    if preset == 'XZ':
        return ['X', 'Z', 'Y']
    return ['Y', 'Z', 'X']


def families_from_manual_axes(lines, scene):
    families = []
    for axis in ('X', 'Y', 'Z'):
        axis_lines = [line for line in lines if line['axis'] == axis]
        if len(axis_lines) < 2:
            continue
        vp = solve_vp_from_lines(axis_lines)
        if vp:
            families.append({
                'axis': axis,
                'vp': vp,
                'lines': axis_lines,
                'residual': line_residual(vp, axis_lines),
            })
    return families


def sensor_focal_unit(cam, scene):
    aspect = scene.render.resolution_x / max(scene.render.resolution_y, 1)
    sensor_extent = cam.data.sensor_width if aspect > 1 else cam.data.sensor_width / max(aspect, 1e-8)
    return 2.0 * cam.data.lens / max(sensor_extent, 1e-8)


def focal_mm_from_unit(f_unit, cam, scene):
    aspect = scene.render.resolution_x / max(scene.render.resolution_y, 1)
    sensor_extent = cam.data.sensor_width if aspect > 1 else cam.data.sensor_width / max(aspect, 1e-8)
    return 0.5 * f_unit * sensor_extent


def compute_focal_from_vps(vp1, vp2, principal=(0.0, 0.0)):
    d = normalize2((vp1[0] - vp2[0], vp1[1] - vp2[1]))
    proj = dot2(d, (principal[0] - vp2[0], principal[1] - vp2[1]))
    p_uv = (proj * d[0] + vp2[0], proj * d[1] + vp2[1])
    pp_uv = length2((principal[0] - p_uv[0], principal[1] - p_uv[1]))
    fv_puv = length2((vp2[0] - p_uv[0], vp2[1] - p_uv[1]))
    fu_puv = length2((vp1[0] - p_uv[0], vp1[1] - p_uv[1]))
    f_sq = fv_puv * fu_puv - pp_uv * pp_uv
    if f_sq <= 0:
        return None
    return math.sqrt(f_sq)


def vp_distance_from_principal(vp, principal=(0.0, 0.0)):
    return length2((vp[0] - principal[0], vp[1] - principal[1]))


def has_far_vanishing_point(vps, principal=(0.0, 0.0), threshold=18.0):
    return any(vp_distance_from_principal(vp, principal) > threshold for vp in vps)


def focal_vp_pair(scene, families, vps_by_axis):
    axis_a, axis_b = axis_names_from_preset(scene)
    if axis_a in vps_by_axis and axis_b in vps_by_axis:
        return vps_by_axis[axis_a], vps_by_axis[axis_b]
    return families[0]['vp'], families[1]['vp']


def compute_principal_and_focal_from_three_vps(vps_by_axis):
    axes = [axis for axis in ('X', 'Y', 'Z') if axis in vps_by_axis]
    if len(axes) < 3:
        return None, None
    pairs = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
    rows, rhs = [], []
    for a, b in pairs:
        v1, v2 = vps_by_axis[a], vps_by_axis[b]
        rows.append((-(v1[0] + v2[0]), -(v1[1] + v2[1]), 1.0))
        rhs.append(-(v1[0] * v2[0] + v1[1] * v2[1]))
    try:
        solution = mathutils.Matrix(rows).inverted() @ mathutils.Vector(rhs)
    except:
        return None, None
    principal = (solution.x, solution.y)
    f_sq = solution.z - principal[0] * principal[0] - principal[1] * principal[1]
    if f_sq <= 1e-8 or f_sq > 10000.0:
        return None, None
    return principal, math.sqrt(f_sq)


def compute_rotation(vp1, vp2, f_unit, principal=(0.0, 0.0)):
    x_axis = mathutils.Vector((vp1[0] - principal[0], vp1[1] - principal[1], -f_unit)).normalized()
    y_axis = mathutils.Vector((vp2[0] - principal[0], vp2[1] - principal[1], -f_unit)).normalized()
    z_axis = x_axis.cross(y_axis)
    if z_axis.length < 1e-8:
        return None
    z_axis.normalize()
    y_axis = z_axis.cross(x_axis).normalized()
    mat = mathutils.Matrix([
        [x_axis.x, y_axis.x, z_axis.x],
        [x_axis.y, y_axis.y, z_axis.y],
        [x_axis.z, y_axis.z, z_axis.z],
    ]).transposed()
    return mat.to_4x4()


def axis_direction_from_vp(vp, principal, f_unit):
    return mathutils.Vector((vp[0] - principal[0], vp[1] - principal[1], -f_unit)).normalized()


def camera_world_rotation_from_axis_vps(vps_by_axis, principal, f_unit):
    axis_dirs = {}
    for axis, vp in vps_by_axis.items():
        if axis in {'X', 'Y', 'Z'}:
            axis_dirs[axis] = axis_direction_from_vp(vp, principal, f_unit)

    if 'X' in axis_dirs and 'Y' in axis_dirs and 'Z' not in axis_dirs:
        axis_dirs['Z'] = axis_dirs['X'].cross(axis_dirs['Y']).normalized()
    if 'X' in axis_dirs and 'Z' in axis_dirs and 'Y' not in axis_dirs:
        axis_dirs['Y'] = axis_dirs['Z'].cross(axis_dirs['X']).normalized()
    if 'Y' in axis_dirs and 'Z' in axis_dirs and 'X' not in axis_dirs:
        axis_dirs['X'] = axis_dirs['Y'].cross(axis_dirs['Z']).normalized()

    if not all(axis in axis_dirs for axis in ('X', 'Y', 'Z')):
        return None

    world_to_camera = mathutils.Matrix((
        (axis_dirs['X'].x, axis_dirs['Y'].x, axis_dirs['Z'].x),
        (axis_dirs['X'].y, axis_dirs['Y'].y, axis_dirs['Z'].y),
        (axis_dirs['X'].z, axis_dirs['Y'].z, axis_dirs['Z'].z),
    ))
    if world_to_camera.determinant() < 0:
        axis_dirs['Z'] = -axis_dirs['Z']
        world_to_camera = mathutils.Matrix((
            (axis_dirs['X'].x, axis_dirs['Y'].x, axis_dirs['Z'].x),
            (axis_dirs['X'].y, axis_dirs['Y'].y, axis_dirs['Z'].y),
            (axis_dirs['X'].z, axis_dirs['Y'].z, axis_dirs['Z'].z),
        ))
    return world_to_camera.transposed().to_4x4()


def image_plane_ray_camera(point, principal, f_unit):
    return mathutils.Vector((point[0] - principal[0], point[1] - principal[1], -f_unit)).normalized()


def place_camera_from_origin(cam, scene, principal, f_unit):
    origin_point = relative_to_image_plane(scene.reblam_lines_origin_x, scene.reblam_lines_origin_y, scene)
    ray_cam = image_plane_ray_camera(origin_point, principal, f_unit)
    ray_world = (cam.matrix_world.to_3x3() @ ray_cam).normalized()
    cam.location = -ray_world * scene.reblam_lines_camera_distance


def align_coordinate_axes(mat, ax1, ax2):
    mat = mathutils.Euler((math.radians(180), 0, 0)).to_matrix().to_4x4() @ mat @ mathutils.Euler((0, 0, math.radians(180))).to_matrix().to_4x4()
    if ax1 == 1 and ax2 == 0:
        mat = mathutils.Euler((0, 0, math.radians(90))).to_matrix().to_4x4() @ mat
    elif ax1 == 0 and ax2 == 2:
        mat = mathutils.Euler((math.radians(-90), 0, 0)).to_matrix().to_4x4() @ mat
    elif ax1 == 2 and ax2 == 0:
        mat = mathutils.Euler((math.radians(-90), 0, 0)).to_matrix().to_4x4() @ mathutils.Euler((0, 0, math.radians(-90))).to_matrix().to_4x4() @ mat
    elif ax1 == 1 and ax2 == 2:
        mat = mathutils.Euler((0, math.radians(-90), 0)).to_matrix().to_4x4() @ mathutils.Euler((0, 0, math.radians(90))).to_matrix().to_4x4() @ mat
    elif ax1 == 2 and ax2 == 1:
        mat = mathutils.Euler((0, math.radians(-90), 0)).to_matrix().to_4x4() @ mat
    return mat


def axis_pair_from_preset(scene):
    preset = scene.reblam_lines_axis_preset
    if preset == 'XY':
        return 0, 1
    if preset == 'XZ':
        return 0, 2
    return 1, 2


def axis_names_from_preset(scene):
    preset = scene.reblam_lines_axis_preset
    if preset == 'XY':
        return 'X', 'Y'
    if preset == 'XZ':
        return 'X', 'Z'
    return 'Y', 'Z'


def axis_index(axis):
    return {'X': 0, 'Y': 1, 'Z': 2}[axis]


def plane_normal_axis(axis_a, axis_b):
    used = {axis_index(axis_a), axis_index(axis_b)}
    return next(index for index in (0, 1, 2) if index not in used)


def plane_normal_vector(scene):
    axis_a, axis_b = axis_names_from_preset(scene)
    normal_index = plane_normal_axis(axis_a, axis_b)
    normal = mathutils.Vector((0.0, 0.0, 0.0))
    normal[normal_index] = 1.0
    return normal, normal_index


def plane_widget_center(scene, cam):
    normal, normal_index = plane_normal_vector(scene)
    plane_value = getattr(scene, "reblam_lines_plane_offset", 0.0)
    hit = camera_uv_ray_plane_intersection(scene, cam, 0.5, 0.5, normal_index, plane_value)
    if hit:
        return hit
    center = mathutils.Vector((0.0, 0.0, 0.0))
    center[normal_index] = plane_value
    return center + normal * max(scene.reblam_lines_camera_distance * 0.15, 1.0)


def camera_uv_ray(scene, cam, u, v):
    target = uv_to_world(scene, cam, u, v)
    origin = cam.location
    direction = (target - origin).normalized()
    return origin, direction


def camera_uv_ray_plane_intersection(scene, cam, u, v, normal_index, plane_value=0.0):
    origin, direction = camera_uv_ray(scene, cam, u, v)
    denom = direction[normal_index]
    if abs(denom) < 1e-8:
        return None
    distance = (plane_value - origin[normal_index]) / denom
    if distance <= 0:
        return None
    return origin + direction * distance


def camera_background_image(cam):
    for bg in cam.data.background_images:
        if bg.source == 'IMAGE' and bg.image:
            return bg.image
    return None


def apply_photo_material(obj, image, uv_name):
    if not image:
        return None
    material = bpy.data.materials.new(f"{obj.name}_Projected_Texture")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = uv_name
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.image = image
    image_node.extension = 'CLIP'
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(uv_node.outputs["UV"], image_node.inputs["Vector"])
    material.node_tree.links.new(image_node.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs[0], output.inputs[0])
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return material


def ensure_uv_layer(obj, uv_name):
    if obj.type != 'MESH':
        return None
    mesh = obj.data
    uv_layer = mesh.uv_layers.get(uv_name) or mesh.uv_layers.new(name=uv_name)
    mesh.uv_layers.active = uv_layer
    return uv_layer


def add_live_projection_modifier(scene, cam, obj, image, uv_name):
    if not ensure_uv_layer(obj, uv_name):
        return None
    old_modifier = obj.modifiers.get("reBlam Live Camera Projection")
    if old_modifier:
        obj.modifiers.remove(old_modifier)
    modifier = obj.modifiers.new("reBlam Live Camera Projection", 'UV_PROJECT')
    modifier.uv_layer = uv_name
    if hasattr(modifier, "projector_count"):
        modifier.projector_count = 1
    if modifier.projectors:
        modifier.projectors[0].object = cam
    if image and image.size[0] and image.size[1]:
        if hasattr(modifier, "aspect_x"):
            modifier.aspect_x = image.size[0]
        if hasattr(modifier, "aspect_y"):
            modifier.aspect_y = image.size[1]
    if hasattr(modifier, "image"):
        modifier.image = image
    if hasattr(modifier, "show_in_editmode"):
        modifier.show_in_editmode = True
    return modifier


def apply_live_projection_modifier(context, obj):
    modifier = obj.modifiers.get("reBlam Live Camera Projection")
    if not modifier:
        return False
    active = context.view_layer.objects.active
    was_selected = obj.select_get()
    mode = obj.mode
    try:
        if mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for item in context.selected_objects:
            item.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        return True
    except Exception:
        return False
    finally:
        obj.select_set(was_selected)
        context.view_layer.objects.active = active
        if mode != 'OBJECT' and context.view_layer.objects.active:
            try:
                bpy.ops.object.mode_set(mode=mode)
            except Exception:
                pass


def assign_camera_projected_uvs(scene, cam, obj, uv_name="reBlam_CameraProjection"):
    if obj.type != 'MESH':
        return False
    uv_layer = ensure_uv_layer(obj, uv_name)
    mesh = obj.data
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vertex = mesh.vertices[mesh.loops[loop_index].vertex_index]
            world = obj.matrix_world @ vertex.co
            projected = world_to_camera_view(scene, cam, world)
            uv_layer.data[loop_index].uv = (projected.x, projected.y)
    return True


def create_projected_plane_mesh(context):
    scene = context.scene
    cam = scene.camera
    subdivisions = max(1, scene.reblam_lines_plane_subdivisions)
    axis_a, axis_b, ax_a, ax_b, normal_index, plane_value, min_a, max_a, min_b, max_b, has_hits = visible_plane_bounds(scene, cam)
    if not has_hits:
        return None, "The current preview plane is not visible in front of the camera. Move Plane Height or choose another plane orientation."

    vertices = []
    for y in range(subdivisions + 1):
        t = y / subdivisions
        value_b = min_b + (max_b - min_b) * t
        for x in range(subdivisions + 1):
            s = x / subdivisions
            value_a = min_a + (max_a - min_a) * s
            coords = [0.0, 0.0, 0.0]
            coords[ax_a] = value_a
            coords[ax_b] = value_b
            coords[normal_index] = plane_value
            vertices.append(tuple(coords))

    faces = []
    stride = subdivisions + 1
    for y in range(subdivisions):
        for x in range(subdivisions):
            i = y * stride + x
            faces.append((i, i + 1, i + stride + 1, i + stride))

    mesh = bpy.data.meshes.new(f"reBlam_{scene.reblam_lines_axis_preset}_Projected_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"reBlam {scene.reblam_lines_axis_preset} Projected Plane", mesh)
    context.collection.objects.link(obj)
    context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj, None


def camera_shift_from_principal(principal, scene):
    return -0.5 * principal[0], -0.5 * principal[1]


def nearest_guide_handle(scene, cam, region, region_data, mouse_xy):
    best = None
    best_dist = 28.0
    for index, guide in enumerate(scene.reblam_lines):
        p1 = uv_to_region(scene, cam, region, region_data, guide.x1, guide.y1)
        p2 = uv_to_region(scene, cam, region, region_data, guide.x2, guide.y2)
        if p1 is None or p2 is None:
            continue
        for endpoint, point in enumerate((p1, p2)):
            dist = length2((point[0] - mouse_xy[0], point[1] - mouse_xy[1]))
            if dist < best_dist:
                best = (index, endpoint)
                best_dist = dist
    return best


def distance_to_segment(point, a, b):
    ab = (b[0] - a[0], b[1] - a[1])
    ap = (point[0] - a[0], point[1] - a[1])
    denom = max(dot2(ab, ab), 1e-8)
    t = clamp(dot2(ap, ab) / denom)
    closest = (a[0] + ab[0] * t, a[1] + ab[1] * t)
    return length2((point[0] - closest[0], point[1] - closest[1]))


def nearest_guide_line(scene, cam, region, region_data, mouse_xy):
    best_index = -1
    best_dist = 16.0
    for index, guide in enumerate(scene.reblam_lines):
        p1 = uv_to_region(scene, cam, region, region_data, guide.x1, guide.y1)
        p2 = uv_to_region(scene, cam, region, region_data, guide.x2, guide.y2)
        if p1 is None or p2 is None:
            continue
        dist = distance_to_segment(mouse_xy, p1, p2)
        if dist < best_dist:
            best_index = index
            best_dist = dist
    return best_index


class REBLAMLinesGuide(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Name", default="Line")
    x1: bpy.props.FloatProperty(name="X1", default=0.25, min=0.0, max=1.0)
    y1: bpy.props.FloatProperty(name="Y1", default=0.5, min=0.0, max=1.0)
    x2: bpy.props.FloatProperty(name="X2", default=0.75, min=0.0, max=1.0)
    y2: bpy.props.FloatProperty(name="Y2", default=0.5, min=0.0, max=1.0)
    axis: bpy.props.EnumProperty(
        name="Axis",
        description="Perspective axis represented by this guide line",
        items=[
            ('AUTO', "Auto", "Let Analyze assign an axis"),
            ('X', "X", "World X axis"),
            ('Y', "Y", "World Y axis"),
            ('Z', "Z", "World Z axis"),
        ],
        default='AUTO'
    )
    family: bpy.props.IntProperty(name="Family", default=-1)


class REBLAMLINES_OT_load_image(bpy.types.Operator):
    bl_idname = "reblam_lines.load_image"
    bl_label = "Load Image"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        try:
            image = bpy.data.images.load(self.filepath, check_existing=True)
        except Exception as exc:
            self.report({'ERROR'}, f"Could not load image: {exc}")
            return {'CANCELLED'}

        scene = context.scene
        scene.render.resolution_x = image.size[0]
        scene.render.resolution_y = image.size[1]

        cam = scene.camera
        if not cam:
            cam_data = bpy.data.cameras.new("reBlam_Lines_Camera")
            cam = bpy.data.objects.new("reBlam_Lines_Camera", cam_data)
            context.collection.objects.link(cam)
            scene.camera = cam
            cam.location = (0, -10, 0)
            cam.rotation_euler = (math.radians(90), 0, 0)

        cam.data.show_background_images = True
        cam.data.background_images.clear()
        bg = cam.data.background_images.new()
        bg.source = 'IMAGE'
        bg.image = image
        bg.alpha = 0.55
        bg.frame_method = 'CROP'
        ensure_draw_handler()

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces[0].region_3d.view_perspective = 'CAMERA'
                area.spaces[0].shading.type = 'WIREFRAME'

        start_line_editor(context)
        self.report({'INFO'}, "Image loaded. Add at least four custom guide lines.")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class REBLAMLINES_OT_add_line(bpy.types.Operator):
    bl_idname = "reblam_lines.add_line"
    bl_label = "Add Line"
    bl_description = "Add a custom drawn perspective guide line."
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        index = len(scene.reblam_lines)
        guide = scene.reblam_lines.add()
        guide.name = f"Line {index + 1}"
        offset = ((index % 7) - 3) * 0.055
        guide.x1 = 0.22
        guide.y1 = clamp(0.5 + offset)
        guide.x2 = 0.78
        guide.y2 = clamp(0.5 + offset)
        if index > 0:
            guide.axis = scene.reblam_lines[index - 1].axis
        else:
            guide.axis = 'X'
        guide.family = -1
        scene.reblam_lines_active_index = index
        ensure_draw_handler()
        start_line_editor(context)
        self.report({'INFO'}, "Guide line added. Drag its endpoints directly in the viewport.")
        return {'FINISHED'}


class REBLAMLINES_OT_edit_lines(bpy.types.Operator):
    bl_idname = "reblam_lines.edit_lines"
    bl_label = "Edit Lines"
    bl_description = "Click and drag the nearest custom guide endpoint in the camera view."

    _drag = None
    _last_uv = None

    def modal(self, context, event):
        global _line_editor_running
        scene = context.scene
        cam = scene.camera
        if event.type in {'ESC', 'RIGHTMOUSE'} and self._drag:
            self._drag = None
            self._last_uv = None
            return {'RUNNING_MODAL'}
        if not cam or context.area.type != 'VIEW_3D':
            _line_editor_running = False
            return {'PASS_THROUGH'}

        region_data = context.space_data.region_3d
        mouse_xy = (event.mouse_region_x, event.mouse_region_y)

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            hit = nearest_guide_handle(scene, cam, context.region, region_data, mouse_xy)
            if hit:
                self._drag = hit
                scene.reblam_lines_active_index = hit[0]
                self._last_uv = region_to_uv(scene, cam, context.region, region_data, event.mouse_region_x, event.mouse_region_y)
                return {'RUNNING_MODAL'}
            line_index = nearest_guide_line(scene, cam, context.region, region_data, mouse_xy)
            if line_index >= 0:
                scene.reblam_lines_active_index = line_index
                self._drag = (line_index, 2)
                self._last_uv = region_to_uv(scene, cam, context.region, region_data, event.mouse_region_x, event.mouse_region_y)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
        elif event.type == 'MOUSEMOVE' and self._drag:
            uv = region_to_uv(scene, cam, context.region, region_data, event.mouse_region_x, event.mouse_region_y)
            if uv:
                guide = scene.reblam_lines[self._drag[0]]
                if self._drag[1] == 0:
                    guide.x1, guide.y1 = uv
                elif self._drag[1] == 1:
                    guide.x2, guide.y2 = uv
                elif self._last_uv:
                    dx = uv[0] - self._last_uv[0]
                    dy = uv[1] - self._last_uv[1]
                    guide.x1 = clamp(guide.x1 + dx)
                    guide.y1 = clamp(guide.y1 + dy)
                    guide.x2 = clamp(guide.x2 + dx)
                    guide.y2 = clamp(guide.y2 + dy)
                if guide.axis == 'AUTO':
                    guide.family = -1
                self._last_uv = uv
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if self._drag:
                self._drag = None
                self._last_uv = None
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        global _line_editor_running
        if _line_editor_running:
            return {'CANCELLED'}
        if not context.scene.camera:
            self.report({'ERROR'}, "Load an image or create a camera first.")
            return {'CANCELLED'}
        if context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run Edit Lines from a 3D View.")
            return {'CANCELLED'}
        ensure_draw_handler()
        _line_editor_running = True
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class REBLAMLINES_OT_analyze_solve(bpy.types.Operator):
    bl_idname = "reblam_lines.analyze_solve"
    bl_label = "Analyze Lines & Solve"
    bl_description = "Infer perspective line families, estimate vanishing points, and solve the camera."
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        cam = scene.camera
        if not cam:
            self.report({'ERROR'}, "No scene camera.")
            return {'CANCELLED'}
        if len(scene.reblam_lines) < 4:
            self.report({'ERROR'}, "Add at least four guide lines.")
            return {'CANCELLED'}

        lines = collect_projected_lines(scene)
        threshold = scene.reblam_lines_inlier_threshold
        axis_order = axis_order_for_preset(scene)
        manual_families = families_from_manual_axes(lines, scene)
        families = manual_families
        if len(families) < 2:
            families = infer_vanishing_families(lines, threshold, scene.reblam_lines_max_families)
        classify_lines(scene, families, axis_order)
        if len(families) < 2:
            self.report({'ERROR'}, "Could not infer two vanishing point families. Add or adjust lines.")
            return {'CANCELLED'}

        families.sort(key=lambda f: (len(f['lines']), -f['residual']), reverse=True)
        vps_by_axis = {family.get('axis'): family['vp'] for family in families if family.get('axis') in {'X', 'Y', 'Z'}}
        principal = (0.0, 0.0)
        used_three_vp_calibration = False
        far_vp_present = has_far_vanishing_point(vps_by_axis.values(), principal)
        if scene.reblam_lines_use_principal_solve and len(vps_by_axis) >= 3 and not far_vp_present:
            solved_principal, solved_focal = compute_principal_and_focal_from_three_vps(vps_by_axis)
            if solved_principal and solved_focal:
                principal = solved_principal
                f_unit = solved_focal
                used_three_vp_calibration = True
            else:
                f_unit = None
        else:
            f_unit = None

        vp1, vp2 = focal_vp_pair(scene, families, vps_by_axis)

        used_current_lens = False
        if f_unit is None and not has_far_vanishing_point((vp1, vp2), principal):
            f_unit = compute_focal_from_vps(vp1, vp2, principal)
        if f_unit is None or f_unit > 100.0:
            f_unit = sensor_focal_unit(cam, scene)
            used_current_lens = True

        rotation = camera_world_rotation_from_axis_vps(vps_by_axis, principal, f_unit)
        if rotation is None:
            self.report({'ERROR'}, "Could not compute camera rotation from assigned VP axes.")
            return {'CANCELLED'}

        cam.matrix_world = rotation
        cam.data.sensor_fit = 'HORIZONTAL'
        cam.data.lens = focal_mm_from_unit(f_unit, cam, scene)
        if used_three_vp_calibration:
            cam.data.shift_x, cam.data.shift_y = camera_shift_from_principal(principal, scene)
        else:
            cam.data.shift_x = 0
            cam.data.shift_y = 0
        place_camera_from_origin(cam, scene, principal, f_unit)
        context.view_layer.update()

        family_msg = ", ".join([f"{fam.get('axis', '?')}:{len(fam['lines'])}" for fam in families])
        msg = f"Inferred {len(families)} families ({family_msg}). FL: {cam.data.lens:.2f}mm"
        if used_three_vp_calibration:
            msg += " (3-VP principal solve)"
        if used_current_lens:
            msg += " (kept current focal length)"
        if far_vp_present:
            msg += " (far VP: skipped optical-center solve)"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class REBLAMLINES_OT_create_projected_plane(bpy.types.Operator):
    bl_idname = "reblam_lines.create_projected_plane"
    bl_label = "Create Textured Plane"
    bl_description = "Create a real floor or wall mesh from the selected line axes and camera-project the loaded image onto it."
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        cam = scene.camera
        if not cam:
            self.report({'ERROR'}, "No scene camera.")
            return {'CANCELLED'}

        obj, error = create_projected_plane_mesh(context)
        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}

        image = camera_background_image(cam)
        if not image:
            self.report({'ERROR'}, "No camera background image to project.")
            return {'CANCELLED'}

        uv_name = "reBlam_CameraProjection"
        apply_photo_material(obj, image, uv_name)
        if not add_live_projection_modifier(scene, cam, obj, image, uv_name):
            self.report({'ERROR'}, "Could not add live camera projection.")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Created live-projected {scene.reblam_lines_axis_preset} plane. Click Finalize Texture to bake UVs.")
        return {'FINISHED'}


class REBLAMLINES_OT_finalize_texture(bpy.types.Operator):
    bl_idname = "reblam_lines.finalize_texture"
    bl_label = "Finalize Texture"
    bl_description = "Write the current camera projection into mesh UVs and assign a normal image material."
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        cam = scene.camera
        obj = context.active_object
        if not cam:
            self.report({'ERROR'}, "No scene camera.")
            return {'CANCELLED'}
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh plane to finalize.")
            return {'CANCELLED'}

        image = camera_background_image(cam)
        if not image:
            self.report({'ERROR'}, "No camera background image to project.")
            return {'CANCELLED'}

        uv_name = "reBlam_CameraProjection"
        applied_live = apply_live_projection_modifier(context, obj)
        if not applied_live and not assign_camera_projected_uvs(scene, cam, obj, uv_name):
            self.report({'ERROR'}, "Could not write projected UVs.")
            return {'CANCELLED'}
        apply_photo_material(obj, image, uv_name)
        self.report({'INFO'}, "Texture finalized into permanent mesh UVs and material.")
        return {'FINISHED'}


class REBLAMLINES_OT_clear_lines(bpy.types.Operator):
    bl_idname = "reblam_lines.clear_lines"
    bl_label = "Clear Lines"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        context.scene.reblam_lines.clear()
        context.scene.reblam_lines_active_index = -1
        self.report({'INFO'}, "Guide lines cleared.")
        return {'FINISHED'}


class REBLAMLINES_UL_guides(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        axis_label = item.axis if item.axis != 'AUTO' else "Auto"
        layout.label(text=f"{item.name}  {axis_label}", icon='IPO_LINEAR')


class REBLAMLINES_GGT_plane_height(bpy.types.GizmoGroup):
    bl_idname = "REBLAMLINES_GGT_plane_height"
    bl_label = "reBlam Plane Height"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {'3D', 'PERSISTENT'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return (
            scene
            and scene.camera
            and hasattr(scene, "reblam_lines_plane_offset")
            and getattr(scene, "reblam_lines_show_plane_grid", False)
        )

    def setup(self, context):
        gizmo = self.gizmos.new("GIZMO_GT_arrow_3d")
        gizmo.target_set_prop("offset", context.scene, "reblam_lines_plane_offset")
        gizmo.color = (1.0, 0.9, 0.15)
        gizmo.alpha = 0.55
        gizmo.color_highlight = (1.0, 1.0, 0.25)
        gizmo.alpha_highlight = 0.95
        gizmo.scale_basis = 1.4
        self.plane_height_gizmo = gizmo

    def draw_prepare(self, context):
        scene = context.scene
        cam = scene.camera
        if not cam:
            return
        normal, normal_index = plane_normal_vector(scene)
        center = plane_widget_center(scene, cam)
        center[normal_index] = 0.0
        rotation = normal.to_track_quat('Z', 'Y').to_matrix().to_4x4()
        self.plane_height_gizmo.matrix_basis = mathutils.Matrix.Translation(center) @ rotation


class REBLAMLINES_PT_panel(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_reblam_lines"
    bl_label = "reBlam Lines"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "reBlam Lines"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.operator("reblam_lines.load_image", icon='FILE_IMAGE')
        row = layout.row(align=True)
        row.operator("reblam_lines.add_line", icon='ADD')
        row = layout.row(align=True)
        row.operator("reblam_lines.analyze_solve", icon='OUTLINER_OB_CAMERA')
        row.operator("reblam_lines.create_projected_plane")
        row = layout.row(align=True)
        row.operator("reblam_lines.finalize_texture")
        row.operator("reblam_lines.clear_lines", icon='TRASH')

        box = layout.box()
        box.prop(scene, "reblam_lines_axis_preset", text="Plane")
        box.prop(scene, "reblam_lines_plane_offset", text="Plane Height")
        box.prop(scene, "reblam_lines_plane_extent", text="Plane Extent")
        box.prop(scene, "reblam_lines_plane_subdivisions", text="Plane Detail")
        box.prop(scene, "reblam_lines_inlier_threshold", text="Line Tolerance")
        box.prop(scene, "reblam_lines_max_families", text="Max Families")
        box.prop(scene, "reblam_lines_use_principal_solve", text="3-Axis Optical Center")
        box.prop(scene, "reblam_lines_origin_x", text="Origin X")
        box.prop(scene, "reblam_lines_origin_y", text="Origin Y")
        box.prop(scene, "reblam_lines_camera_distance", text="Camera Distance")
        box.prop(scene, "reblam_lines_show_guides", text="Show Guides")
        box.prop(scene, "reblam_lines_show_plane_grid", text="Show Plane Grid")
        box.prop(scene, "reblam_lines_width", text="Line Width")
        box.prop(scene, "reblam_lines_grid_width", text="Grid Width")
        box.prop(scene, "reblam_lines_grid_opacity", text="Grid Opacity")
        box.label(text=f"Guide Lines: {len(scene.reblam_lines)} / 4 minimum")
        box.template_list(
            "REBLAMLINES_UL_guides",
            "",
            scene,
            "reblam_lines",
            scene,
            "reblam_lines_active_index",
            rows=4
        )
        active = scene.reblam_lines_active_index
        if 0 <= active < len(scene.reblam_lines):
            guide = scene.reblam_lines[active]
            axis_box = box.box()
            axis_box.prop(guide, "axis", text="Active Axis")
            axis_box.prop(guide, "name", text="Name")


classes = (
    REBLAMLinesGuide,
    REBLAMLINES_OT_load_image,
    REBLAMLINES_OT_add_line,
    REBLAMLINES_OT_edit_lines,
    REBLAMLINES_OT_analyze_solve,
    REBLAMLINES_OT_create_projected_plane,
    REBLAMLINES_OT_finalize_texture,
    REBLAMLINES_OT_clear_lines,
    REBLAMLINES_UL_guides,
    REBLAMLINES_GGT_plane_height,
    REBLAMLINES_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.reblam_lines = bpy.props.CollectionProperty(type=REBLAMLinesGuide)
    bpy.types.Scene.reblam_lines_active_index = bpy.props.IntProperty(name="Active Guide", default=-1)
    bpy.types.Scene.reblam_lines_axis_preset = bpy.props.EnumProperty(
        name="Plane",
        description="Choose the preview world plane to grid, move, and texture",
        items=[
            ('XY', "Floor (XY)", "Create and preview a horizontal floor plane. X/Z guide lines can still be used to solve the camera."),
            ('XZ', "Wall XZ", "Create and preview a vertical wall plane spanning X and Z"),
            ('YZ', "Wall YZ", "Create and preview a vertical wall plane spanning Y and Z"),
        ],
        default='XY'
    )
    bpy.types.Scene.reblam_lines_plane_subdivisions = bpy.props.IntProperty(
        name="Plane Detail",
        description="Subdivisions for the projected floor or wall mesh",
        default=48,
        min=1,
        max=240
    )
    bpy.types.Scene.reblam_lines_plane_extent = bpy.props.FloatProperty(
        name="Plane Extent",
        description="Scale the generated plane bounds relative to the camera-visible plane area. Values above 1 extend beyond the photo and may show clipped texture.",
        default=1.0,
        min=0.25,
        max=3.0,
        precision=2
    )
    bpy.types.Scene.reblam_lines_plane_offset = bpy.props.FloatProperty(
        name="Plane Height",
        description="Offset of the preview floor or wall along its normal axis",
        default=0.0,
        precision=3
    )
    bpy.types.Scene.reblam_lines_inlier_threshold = bpy.props.FloatProperty(
        name="Line Tolerance",
        description="Maximum normalized image-plane distance for a line to join a vanishing family",
        default=0.018,
        min=0.002,
        max=0.12,
        precision=3
    )
    bpy.types.Scene.reblam_lines_max_families = bpy.props.IntProperty(
        name="Max Families",
        description="Maximum number of vanishing point families to classify",
        default=3,
        min=2,
        max=4
    )
    bpy.types.Scene.reblam_lines_use_principal_solve = bpy.props.BoolProperty(
        name="3-Axis Optical Center",
        description="Use three axis vanishing points to estimate optical center and focal length together",
        default=True
    )
    bpy.types.Scene.reblam_lines_origin_x = bpy.props.FloatProperty(
        name="Origin X",
        description="Image-space X coordinate of the world origin",
        default=0.5,
        min=0.0,
        max=1.0
    )
    bpy.types.Scene.reblam_lines_origin_y = bpy.props.FloatProperty(
        name="Origin Y",
        description="Image-space Y coordinate of the world origin",
        default=0.5,
        min=0.0,
        max=1.0
    )
    bpy.types.Scene.reblam_lines_camera_distance = bpy.props.FloatProperty(
        name="Camera Distance",
        description="Default distance from camera to the reconstructed world origin",
        default=10.0,
        min=0.1,
        max=1000.0
    )
    bpy.types.Scene.reblam_lines_show_guides = bpy.props.BoolProperty(
        name="Show Guides",
        description="Draw custom thick guide lines in the viewport",
        default=True
    )
    bpy.types.Scene.reblam_lines_show_plane_grid = bpy.props.BoolProperty(
        name="Show Plane Grid",
        description="Draw the current floor or wall plane grid while adjusting its height",
        default=True
    )
    bpy.types.Scene.reblam_lines_grid_width = bpy.props.FloatProperty(
        name="Grid Width",
        description="Viewport width for the floor or wall preview grid",
        default=2.0,
        min=0.5,
        max=10.0
    )
    bpy.types.Scene.reblam_lines_grid_opacity = bpy.props.FloatProperty(
        name="Grid Opacity",
        description="Opacity for the floor or wall preview grid",
        default=0.45,
        min=0.05,
        max=1.0,
        subtype='FACTOR'
    )
    bpy.types.Scene.reblam_lines_width = bpy.props.FloatProperty(
        name="Line Width",
        description="Viewport width for custom guide lines",
        default=4.0,
        min=1.0,
        max=12.0
    )
    ensure_draw_handler()


def unregister():
    remove_draw_handler()
    del bpy.types.Scene.reblam_lines_width
    del bpy.types.Scene.reblam_lines_grid_opacity
    del bpy.types.Scene.reblam_lines_grid_width
    del bpy.types.Scene.reblam_lines_show_plane_grid
    del bpy.types.Scene.reblam_lines_show_guides
    del bpy.types.Scene.reblam_lines_camera_distance
    del bpy.types.Scene.reblam_lines_origin_y
    del bpy.types.Scene.reblam_lines_origin_x
    del bpy.types.Scene.reblam_lines_use_principal_solve
    del bpy.types.Scene.reblam_lines_max_families
    del bpy.types.Scene.reblam_lines_inlier_threshold
    del bpy.types.Scene.reblam_lines_plane_offset
    del bpy.types.Scene.reblam_lines_plane_extent
    del bpy.types.Scene.reblam_lines_plane_subdivisions
    del bpy.types.Scene.reblam_lines_axis_preset
    del bpy.types.Scene.reblam_lines_active_index
    del bpy.types.Scene.reblam_lines
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
