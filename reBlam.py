import bpy
import mathutils
import math
import cmath
import operator
import bmesh
from bpy_extras.object_utils import world_to_camera_view
from functools import reduce

bl_info = {
    'name': 'BLAM - Camera Calibration (Mesh Workflow v1.13)',
    'author': 'Per Gantelius (Fixed for Blender 4.5+)',
    'version': (0, 1, 14),
    'blender': (4, 5, 0),
    'location': '3D View > Sidebar > Photo Modeling Tools',
    'description': 'Reconstruction of 3D geometry and estimation of camera orientation using Mesh Planes.',
    'category': '3D View'
}

# =============================================================================
# MATH LIBRARY
# =============================================================================

def iszero(z): return abs(z) < .000001

def cbrt(x): 
    if x >= 0: return math.pow(x, 1.0/3.0) 
    else: return -math.pow(abs(x), 1.0/3.0)
    
def polar(x, y, deg=0): 
    if deg: return math.hypot(x, y), 180.0 * math.atan2(y, x) / math.pi 
    else: return math.hypot(x, y), math.atan2(y, x)

def quadratic(a, b, c=None): 
    if c: a, b = b / float(a), c / float(a) 
    t = a / 2.0 
    r = t**2 - b 
    if r >= 0: y1 = math.sqrt(r) 
    else: y1 = cmath.sqrt(r) 
    y2 = -y1 
    return y1 - t, y2 - t    

def solveCubic(a, b, c, d):    
    cIn = [a, b, c, d]
    a, b, c = b / float(a), c / float(a), d / float(a) 
    t = a / 3.0 
    p, q = b - 3 * t**2, c - b * t + 2 * t**3 
    u, v = quadratic(q, -(p/3.0)**3) 
    if type(u) == type(0j): 
        r, w = polar(u.real, u.imag) 
        y1 = 2 * cbrt(r) * math.cos(w / 3.0) 
    else: y1 = cbrt(u) + cbrt(v) 
    y2, y3 = quadratic(y1, p + y1**2) 
    x1, x2, x3 = y1 - t, y2 - t, y3 - t
    return x1, x2, x3

class Table(list):
    dim = 1
    concat = list.__add__
    def __getitem__(self, item):
        if isinstance(item, slice): return self.__class__(list.__getitem__(self, item))
        return list.__getitem__(self, item)
    def __init__( self, elems ):
        elems = list(elems)
        list.__init__( self, elems )
        if len(elems) and hasattr(elems[0], 'dim'): self.dim = elems[0].dim + 1
    def map( self, op, rhs=None ):
        if rhs is None: return self.dim==1 and self.__class__( list(map(op, self)) ) or self.__class__( [elem.map(op) for elem in self] )
        elif not hasattr(rhs,'dim'): return self.__class__( [op(e,rhs) for e in self] )
        elif self.dim == rhs.dim: return self.__class__( list(map(op, self, rhs)) )
        elif self.dim < rhs.dim: return self.__class__( [op(self,e) for e in rhs]  )
        return self.__class__( [op(e,rhs) for e in self] )
    def __mul__( self, rhs ):  return self.map( operator.mul, rhs )
    def __div__( self, rhs ):  return self.map( operator.truediv, rhs )
    def __sub__( self, rhs ):  return self.map( operator.sub, rhs )
    def __add__( self, rhs ):  return self.map( operator.add, rhs )
    def __rmul__( self, lhs ):  return self*lhs
    def __rsub__( self, lhs ):  return -(self-lhs)
    def __radd__( self, lhs ):  return self+lhs
    def __abs__( self ): return self.map( abs )
    def __neg__( self ): return self.map( operator.neg )
    def flatten( self ):
        if self.dim == 1: return self
        return reduce( lambda cum, e: e.flatten().concat(cum), self, [] )
    def prod( self ):  return reduce(operator.mul, self.flatten(), 1.0)
    def sum( self ):  return reduce(operator.add, self.flatten(), 0.0)
    def exists( self, predicate ):
        for elem in self.flatten():
            if predicate(elem): return 1
        return 0
    def forall( self, predicate ):
        for elem in self.flatten():
            if not predicate(elem): return 0
        return 1
    def __eq__( self, rhs ):  return (self - rhs).forall( iszero )

class Vec(Table):
    def dot( self, otherVec ):  return reduce(operator.add, list(map(operator.mul, self, otherVec)), 0.0)
    def norm( self ):  return math.sqrt(abs( self.dot(self) ))
    def normalize( self ):  return self * (1.0 / self.norm()) if self.norm() != 0 else self
    def outer( self, otherVec ):  return Mat([otherVec*x for x in self])

class Matrix(Table):
    __slots__ = ['size', 'rows', 'cols']
    def __init__( self, elems ):
        elems = list(elems)
        Table.__init__( self, hasattr(elems[0], 'dot') and elems or list(map(Vec,map(tuple,elems))) )
        self.size = self.rows, self.cols = len(elems), len(elems[0])
    def tr( self ): return Mat(zip(*self))
    def mmul( self, other ):
        if other.dim==2: return Mat( list(map(self.mmul, other.tr())) ).tr()
        return Vec( list(map(other.dot, self)) )
    def qr( self, ROnly=0 ):
        R = self
        m, n = R.size
        for i in range(min(m,n)):
            v, beta = R.tr()[i].house(i)
            R -= v.outer( R.tr().mmul(v)*beta )
        for i in range(1,min(n,m)): R[i][:i] = [0] * i
        R = Mat(R[:n])
        if ROnly: return R
        Q = R.tr().solve(self.tr()).tr()
        return Q, R
    def _solve( self, b ):
        Q, R = self.qr()
        return R.solve( Q.tr().mmul(b) )
    def solve( self, b ):
        if b.dim==2: return Mat( list(map(self.solve, b.tr())) ).tr()
        x = self._solve( b )
        diff = b - self.mmul(x)
        maxdiff = diff.dot(diff)
        for i in range(10):
            xnew = x + self._solve( diff )
            diffnew = b - self.mmul(xnew)
            maxdiffnew = diffnew.dot(diffnew)
            if maxdiffnew >= maxdiff:  break
            x, diff, maxdiff = xnew, diffnew, maxdiffnew
        return x

class Square(Matrix): pass
class Triangular(Square):
    def det( self ):  return self.diag().prod()
class UpperTri(Triangular):
    def _solve( self, b ):
        x = Vec([])
        for i in range(self.rows-1, -1, -1):
            x.insert(0, (b[i] - x.dot(self[i][i+1:])) / self[i][i] )
        return x
class LowerTri(Triangular):
    def _solve( self, b ):
        x = Vec([])
        for i in range(self.rows):
            x.append( (b[i] - x.dot(self[i][:i])) / self[i][i] )
        return x

def Mat( elems ):
    elems = list(elems)
    m, n = len(elems), len(elems[0])
    if m != n: return Matrix(elems)
    for i in range(1, len(elems)):
        if not iszero( max(list(map(abs, elems[i][:i]))) ): break
    else: return UpperTri(elems)
    return LowerTri(elems)

def normalize(vec):
    l = length(vec)
    if l == 0: return [0.0]*len(vec)
    return [x / l for x in vec]

def length(vec): return math.sqrt(sum([x * x for x in vec]))
def dot(x, y): return sum([x[i] * y[i] for i in range(len(x))])

# =============================================================================
# BLAM SOLVER CORE
# =============================================================================

class BlamSolver:
    @staticmethod
    def computeIntersectionPointForLineSegments(lineSet):
        matrixRows, rhsRows = [], []
        fallback_dir = None
        
        for line in lineSet:
            p = line[0]        
            d = [x - y for x, y in zip(line[1], line[0])]
            l = math.sqrt(d[0]**2 + d[1]**2)
            if l == 0: continue
            
            d_norm = [d[0]/l, d[1]/l]
            if fallback_dir is None: fallback_dir = d_norm
            
            n = [d_norm[1], -d_norm[0]]
            matrixRows.append(n)
            rhsRows.append([p[0] * n[0] + p[1] * n[1]])
        
        if len(matrixRows) < 2: return None
        
        det = 0
        if len(matrixRows) == 2 and len(matrixRows[0]) == 2:
             a, b = matrixRows[0][0], matrixRows[0][1]
             c, d = matrixRows[1][0], matrixRows[1][1]
             det = abs(a*d - b*c)
        
        if det < 1e-5:
            inf_dist = 100000.0
            if fallback_dir:
                return [fallback_dir[0] * inf_dist, fallback_dir[1] * inf_dist]
            return [inf_dist, 0]
        
        try: return [f[0] for f in Mat(matrixRows).solve(Mat(rhsRows))]
        except: return [fallback_dir[0] * 100000.0, fallback_dir[1] * 100000.0]

    @staticmethod
    def computeFocalLength(Fu, Fv, P):
        d = normalize([x - y for x, y in zip(Fu, Fv)])
        proj = dot(d, [x - y for x, y in zip(P, Fv)])
        Puv = [proj * x + y for x, y in zip(d, Fv)]
        
        PPuv = length([x - y for x, y in zip(P, Puv)])
        FvPuv = length([x - y for x, y in zip(Fv, Puv)])
        FuPuv = length([x - y for x, y in zip(Fu, Puv)])
        
        fSq = FvPuv * FuPuv - PPuv * PPuv
        if fSq < 0: return None
        return math.sqrt(fSq)
    
    @staticmethod
    def computeCameraRotationMatrix(Fu, Fv, f, P):
        Fu, Fv = list(Fu), list(Fv)
        Fu[0] -= P[0]; Fu[1] -= P[1]
        Fv[0] -= P[0]; Fv[1] -= P[1]
        
        OFu, OFv = [Fu[0], Fu[1], f], [Fv[0], Fv[1], f]
        s1, s2 = length(OFu), length(OFv)
        if s1 < 1e-6 or s2 < 1e-6: return None
            
        u, v = normalize(OFu), normalize(OFv)
        w = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]] 
        
        M = mathutils.Matrix()
        M[0][0], M[0][1], M[0][2] = Fu[0]/s1, Fv[0]/s2, w[0]
        M[1][0], M[1][1], M[1][2] = Fu[1]/s1, Fv[1]/s2, w[1]
        M[2][0], M[2][1], M[2][2] = f/s1, f/s2, w[2]
        M.transpose()
        return M
    
    @staticmethod
    def alignCoordinateAxes(M, ax1, ax2):
        if M is None: return mathutils.Matrix.Identity(4)
        M =  mathutils.Euler((math.radians(180),0,0)).to_matrix().to_4x4() @ M @ mathutils.Euler((0,0,math.radians(180))).to_matrix().to_4x4()
        if ax1==1 and ax2==0: M = mathutils.Euler((0,0,math.radians(90))).to_matrix().to_4x4() @ M
        elif ax1==0 and ax2==2: M = mathutils.Euler((math.radians(-90),0,0)).to_matrix().to_4x4() @ M
        elif ax1==2 and ax2==0: M = mathutils.Euler((math.radians(-90),0,0)).to_matrix().to_4x4() @ mathutils.Euler((0,0,math.radians(-90))).to_matrix().to_4x4() @ M
        elif ax1==1 and ax2==2: M = mathutils.Euler((0,math.radians(-90),0)).to_matrix().to_4x4() @ mathutils.Euler((0,0,math.radians(90))).to_matrix().to_4x4() @ M
        elif ax1==2 and ax2==1: M = mathutils.Euler((0,math.radians(-90),0)).to_matrix().to_4x4() @ M
        return M

# =============================================================================
# OPERATORS
# =============================================================================

class LoadImageAndSetupPlaneOperator(bpy.types.Operator):
    bl_idname = "object.load_image_setup_plane"
    bl_label = "1. Load Image & Add Plane"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    def execute(self, context):
        try: img = bpy.data.images.load(self.filepath, check_existing=True)
        except: return {'CANCELLED'}
        
        scn = context.scene
        scn.render.resolution_x = img.size[0]
        scn.render.resolution_y = img.size[1]
        
        cam_obj = scn.camera
        if not cam_obj:
            cam_data = bpy.data.cameras.new("Calibration_Camera")
            cam_obj = bpy.data.objects.new("Calibration_Camera", cam_data)
            context.collection.objects.link(cam_obj)
            scn.camera = cam_obj
            cam_obj.location = (0, -10, 0)
            cam_obj.rotation_euler = (math.radians(90), 0, 0)

        cam_obj.data.show_background_images = True
        cam_obj.data.background_images.clear()
        bg = cam_obj.data.background_images.new()
        bg.source = 'IMAGE'
        bg.image = img
        bg.alpha = 0.5
        bg.frame_method = 'CROP' 
        
        if context.active_object and context.active_object.mode != 'OBJECT':
             bpy.ops.object.mode_set(mode='OBJECT')
        
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD')
        plane = context.active_object
        plane.name = "Calibration_Plane"
        plane.matrix_world = cam_obj.matrix_world @ mathutils.Matrix.Translation((0, 0, -5))
        
        aspect = img.size[0] / img.size[1]
        if aspect > 1: plane.scale = (aspect, 1, 1)
        else: plane.scale = (1, 1/aspect, 1)
            
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bpy.ops.object.mode_set(mode='EDIT')
        
        # IMPROVEMENT: AUTO WIREFRAME
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces[0].region_3d.view_perspective = 'CAMERA'
                area.spaces[0].shading.type = 'WIREFRAME' # Change to wireframe
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SolveCameraFromMeshOperator(bpy.types.Operator):
    bl_idname = "object.solve_camera_from_mesh"
    bl_label = "2. Reconstruct & Align"
    
    def execute(self, context):
        obj = context.edit_object 
        cam = context.scene.camera
        if not obj or obj.type != 'MESH': return {'CANCELLED'}
        
        bm = bmesh.from_edit_mesh(obj.data)
        verts = [v for v in bm.verts]
        if len(verts) != 4: 
            self.report({'ERROR'}, "Mesh must have exactly 4 vertices")
            return {'CANCELLED'}
        mw = obj.matrix_world
        
        vert_data = [] 
        for v in verts:
            co_3d = mw @ v.co
            co_2d = world_to_camera_view(context.scene, cam, co_3d)
            vert_data.append( {'v_idx': v.index, '3d': co_3d, '2d': co_2d} )
            
        cx = sum([v['2d'].x for v in vert_data]) / 4
        cy = sum([v['2d'].y for v in vert_data]) / 4
        vert_data.sort(key=lambda v: math.atan2(v['2d'].y - cy, v['2d'].x - cx))
        p2d = [v['2d'] for v in vert_data]

        if context.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')

        dx_01 = abs(p2d[0].x - p2d[1].x)
        dy_01 = abs(p2d[0].y - p2d[1].y)
        is_horiz = dx_01 > dy_01
        
        if is_horiz:
            vp1_2d_a, vp1_2d_b = (p2d[0], p2d[1]), (p2d[3], p2d[2])
            vp2_2d_a, vp2_2d_b = (p2d[1], p2d[2]), (p2d[0], p2d[3])
        else:
            vp1_2d_a, vp1_2d_b = (p2d[1], p2d[2]), (p2d[0], p2d[3])
            vp2_2d_a, vp2_2d_b = (p2d[0], p2d[1]), (p2d[3], p2d[2])
        
        w, h = context.scene.render.resolution_x, context.scene.render.resolution_y
        def to_solver(pt): return [w/h * (pt.x - 0.5), (pt.y - 0.5)]

        vp1 = BlamSolver.computeIntersectionPointForLineSegments([
            [to_solver(vp1_2d_a[0]), to_solver(vp1_2d_a[1])],
            [to_solver(vp1_2d_b[0]), to_solver(vp1_2d_b[1])]
        ])
        vp2 = BlamSolver.computeIntersectionPointForLineSegments([
            [to_solver(vp2_2d_a[0]), to_solver(vp2_2d_a[1])],
            [to_solver(vp2_2d_b[0]), to_solver(vp2_2d_b[1])]
        ])
        
        if vp1 is None or vp2 is None:
            self.report({'ERROR'}, "Solve Failed: Parallel lines detected."); return {'CANCELLED'}
        
        # IMPROVEMENT: DYNAMIC AXIS ALIGNMENT FROM DROPDOWN
        preset = context.scene.blam_axis_preset
        if preset == 'XY': # Floor
            ax1, ax2 = 0, 1 
        elif preset == 'XZ': # Front Wall
            ax1, ax2 = 0, 2
        else: # YZ Side Wall
            ax1, ax2 = 1, 2

        if vp2[0] < vp1[0]: vp1, vp2 = vp2, vp1; ax1, ax2 = ax2, ax1 
        
        f = BlamSolver.computeFocalLength(vp1, vp2, [0,0])
        used_default_fl = False
        
        if f is None or f > 100.0:
            current_lens = cam.data.lens
            sensor = cam.data.sensor_width
            f = current_lens / sensor
            used_default_fl = True
            
        M = BlamSolver.computeCameraRotationMatrix(vp1, vp2, f, [0,0])
        if M is None:
             self.report({'ERROR'}, "Solve Failed: Rotation Matrix Error."); return {'CANCELLED'}
             
        cam.matrix_world = BlamSolver.alignCoordinateAxes(M, ax1, ax2)
        cam.location = (0, 0, 2)
        cam.data.sensor_fit = 'HORIZONTAL' 
        cam.data.lens = cam.data.sensor_width * f
        cam.data.shift_x = 0; cam.data.shift_y = 0
        
        context.view_layer.update()
        
        # Reproject Vertices
        frame = cam.data.view_frame(scene=context.scene)
        frame_w = [cam.matrix_world @ v for v in frame]
        cam_pos = cam.matrix_world.translation
        
        depth = 5.0
        for i, item in enumerate(vert_data):
             uv = item['2d'] 
             top = frame_w[3].lerp(frame_w[0], uv.x)
             bot = frame_w[2].lerp(frame_w[1], uv.x)
             pt_on_plane = bot.lerp(top, uv.y)
             ray = (pt_on_plane - cam_pos).normalized()
             obj.data.vertices[item['v_idx']].co = obj.matrix_world.inverted() @ (cam_pos + (ray * depth))
             
        obj.data.update()
        
        # Trigger reconstruction and projection
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        
        bpy.ops.object.compute_depth_information()
        bpy.ops.object.project_bg_onto_mesh()
        
        msg = f"Solved! FL: {round(cam.data.lens, 2)}mm"
        if used_default_fl: msg += " (FL Estimation failed, kept current)"
        self.report({'INFO'}, msg)
        return {'FINISHED'}

class ProjectBackgroundImageOntoMeshOperator(bpy.types.Operator):
    bl_idname = "object.project_bg_onto_mesh"    
    bl_label = "Project background image onto mesh"
    
    def execute(self, context):
        mesh = context.active_object
        camera = context.scene.camera
        if not mesh or mesh.type != 'MESH': return {'CANCELLED'}
        
        image_to_use = None
        if camera.data.background_images:
            for bg in camera.data.background_images:
                if bg.source == 'IMAGE' and bg.image:
                    image_to_use = bg.image; break
        
        if not image_to_use: return {'CANCELLED'}

        mat = mesh.data.materials.get("Projected_Material") or bpy.data.materials.new("Projected_Material")
        if not mesh.data.materials: mesh.data.materials.append(mat)
        else: mesh.data.materials[0] = mat
        
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        tex = nodes.new('ShaderNodeTexImage')
        tex.image = image_to_use
        tex.extension = 'CLIP'
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Roughness'].default_value = 1.0
        bsdf.inputs['Specular IOR Level'].default_value = 0.0
        output = nodes.new('ShaderNodeOutputMaterial')
        links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        for m in mesh.modifiers:
            if m.type in ['UV_PROJECT', 'SUBSURF']: mesh.modifiers.remove(m)
        
        # IMPROVEMENT: INCREASED SUBDIVISIONS FOR BETTER PROJECTION
        sub = mesh.modifiers.new(name="Subsurf", type='SUBSURF')
        sub.subdivision_type = 'SIMPLE'; sub.levels = 6 # Increased from 4 to 6
        
        proj_name = f"{mesh.name}_Projector"
        proj = bpy.data.objects.get(proj_name)
        if not proj:
            proj = bpy.data.objects.new(proj_name, bpy.data.cameras.new(name=f"{proj_name}_Data"))
            context.collection.objects.link(proj)
        
        proj.matrix_world = camera.matrix_world
        proj.data.lens = camera.data.lens
        proj.data.sensor_width = camera.data.sensor_width
        proj.data.sensor_height = camera.data.sensor_height
        proj.data.sensor_fit = camera.data.sensor_fit 
        proj.scale = (1,1,1); proj.hide_viewport = True
        
        uv_mod = mesh.modifiers.new(name="UVProject", type='UV_PROJECT')
        uv_mod.projectors[0].object = proj
        uv_mod.aspect_x = context.scene.render.resolution_x / context.scene.render.resolution_y
        uv_mod.aspect_y = 1.0
        if not mesh.data.uv_layers: mesh.data.uv_layers.new()
        uv_mod.uv_layer = mesh.data.uv_layers[0].name
        
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.spaces[0].shading.type = 'MATERIAL'

        return {'FINISHED'}

class Reconstruct3DMeshOperator(bpy.types.Operator):
    bl_idname = "object.compute_depth_information"    
    bl_label = "Reconstruct 3D geometry"
    
    def evalEq17(self, origin, p1, p2):
        a = [x - y for x, y in zip(origin, p1)]
        b = [x - y for x, y in zip(origin, p2)]
        return dot(a, b)
    
    def computeCi(self, Qab, Qac, Qad, Qbc, Qbd, Qcd):        
        self.C4 = Qad * Qbc * Qbd - Qac * Qbd ** 2
        self.C3 = Qab * Qad * Qbd * Qcd - Qad ** 2 * Qcd + Qac * Qad * Qbd ** 2 - Qad ** 2 * Qbc * Qbd - Qbc * Qbd + 2 * Qab * Qac * Qbd - Qab * Qad * Qbc
        self.C2 = -Qab * Qbd * Qcd - Qab ** 2 * Qad * Qcd + 2 * Qad * Qcd + Qad * Qbc * Qbd - 3 * Qab * Qac * Qad * Qbd + Qab * Qad ** 2 * Qbc + Qab * Qbc + Qac * Qad ** 2 - Qab ** 2 * Qac
        self.C1 = Qab ** 2 * Qcd - Qcd + Qab * Qac * Qbd - Qab * Qad * Qbc + 2 * Qab ** 2 * Qac * Qad - 2 * Qac * Qad
        self.C0 = Qac - Qab ** 2 * Qac
        
    def computeBi(self, Qab, Qac, Qad, Qbc, Qbd, Qcd):
        self.B4 = Qbd - Qbd * Qcd ** 2
        self.B3 = 2 * Qad * Qbd * Qcd ** 2 + Qab * Qcd ** 2 + Qac * Qbd * Qcd - Qad * Qbc * Qcd - 2 * Qad * Qbd - Qab
        self.B2 = - Qbd * Qcd ** 2 - Qab * Qad * Qcd ** 2 - 3 * Qac * Qad * Qbd * Qcd + Qad ** 2 * Qbc * Qcd + Qbc * Qcd - Qab * Qac * Qcd + Qad ** 2 * Qbd + Qac * Qad * Qbc + 2 * Qab * Qad
        self.B1 = 2 * Qac * Qbd * Qcd - Qad * Qbc * Qcd + Qab * Qac * Qad * Qcd + Qac ** 2 * Qad * Qbd - Qac * Qad ** 2 * Qbc - Qac * Qbc - Qab * Qad ** 2
        self.B0 = Qac * Qad * Qbc - Qac ** 2 * Qbd

    def worldToCameraSpace(self, verts):
        ret = []
        for v in verts:
            vec = v.co.to_4d()    
            vec = self.mesh.matrix_world @ vec
            vec = self.camera.matrix_world.inverted() @ vec     
            ret.append(vec[0:3])
        return ret
    
    def computeQuadDepthInformation(self, qHatA, qHatB, qHatC, qHatD):
        Qab, Qac, Qad = dot(qHatA, qHatB), dot(qHatA, qHatC), dot(qHatA, qHatD)
        Qba, Qbc, Qbd = dot(qHatB, qHatA), dot(qHatB, qHatC), dot(qHatB, qHatD)
        Qca, Qcb, Qcd = dot(qHatC, qHatA), dot(qHatC, qHatB), dot(qHatC, qHatD)
        self.computeCi(Qab, Qac, Qad, Qbc, Qbd, Qcd)
        self.computeBi(Qab, Qac, Qad, Qbc, Qbd, Qcd)
        if abs(self.B4) < 1e-6: self.B4 = 1e-6
        self.D3 = (self.C4 / self.B4) * self.B3 - self.C3
        self.D2 = (self.C4 / self.B4) * self.B2 - self.C2
        self.D1 = (self.C4 / self.B4) * self.B1 - self.C1
        self.D0 = (self.C4 / self.B4) * self.B0 - self.C0
        roots = solveCubic(self.D3, self.D2, self.D1, self.D0)
        chosenRoot, minError = None, None
        for root in roots:
            if type(root) == type(0j) or root <= 0: continue
            lambdaD = root
            self.lambdaA = 1
            denLambdaA = (Qbd * lambdaD - Qab)
            if abs(denLambdaA) < 1e-6: denLambdaA = 1e-6
            self.lambdaB = (Qad * lambdaD - 1.0) / denLambdaA
            denLambdaC = (Qac - Qcd * lambdaD)
            if abs(denLambdaC) < 1e-6: denLambdaC = 1e-6
            self.lambdaC = (Qad * lambdaD - lambdaD * lambdaD) / denLambdaC
            self.lambdaD = lambdaD
            pA, pB = [x * self.lambdaA for x in qHatA], [x * self.lambdaB for x in qHatB]
            pC, pD = [x * self.lambdaC for x in qHatC], [x * self.lambdaD for x in qHatD]
            meanError, maxError = self.getQuadError(pA, pB, pC, pD)
            if minError == None or meanError < minError:
                minError = meanError; chosenRoot = root
        if chosenRoot == None: chosenRoot = 1.0
        lambdaD = chosenRoot
        self.lambdaA = 1 
        denLambdaA = (Qbd * lambdaD - Qab)
        if abs(denLambdaA) < 1e-6: denLambdaA = 1e-6
        self.lambdaB = (Qad * lambdaD - 1.0) / denLambdaA
        denLambdaC = (Qac - Qcd * lambdaD)
        if abs(denLambdaC) < 1e-6: denLambdaC = 1e-6
        self.lambdaC = (Qad * lambdaD - lambdaD * lambdaD) / denLambdaC
        self.lambdaD = lambdaD
        pA, pB = [x * self.lambdaA for x in qHatA], [x * self.lambdaB for x in qHatB]
        pC, pD = [x * self.lambdaC for x in qHatC], [x * self.lambdaD for x in qHatD]
        return [pA, pB, pC, pD]
    
    def getQuadError(self, pA, pB, pC, pD):
        errs = [abs(self.evalEq17(pA, pB, pD)), abs(self.evalEq17(pB, pA, pC)), 
                abs(self.evalEq17(pC, pB, pD)), abs(self.evalEq17(pD, pA, pC))]
        return 0.25 * sum(errs), max(errs)
               
    def createMesh(self, inputMesh, computedCoordsByFace, quads):
        quadFacePairsBySharedEdge, quadFaces = {}, []
        for e in inputMesh.data.edges:
            facesContainingEdge = [f for f in inputMesh.data.polygons 
                                   if len(f.vertices)==4 and len(set(f.vertices) & set(e.vertices)) == 2]
            if len(facesContainingEdge) == 2:
                quadFacePairsBySharedEdge[e.index] = facesContainingEdge
        numQuadFaces = len(computedCoordsByFace)
        matrixRows, rhRows = [], []
        face_to_idx = {f: i for i, f in enumerate(computedCoordsByFace.keys())}
        for eIdx, pair in quadFacePairsBySharedEdge.items():
            f0, f1 = pair
            f0Idx, f1Idx = face_to_idx[f0], face_to_idx[f1]
            c0, c1 = computedCoordsByFace[f0], computedCoordsByFace[f1]
            edge = inputMesh.data.edges[eIdx]
            def get_depths(quad, v_idx):
                for p in quad:
                    if p[-1] == v_idx: return p[2]
                return 1.0
            l00, l10 = get_depths(c0, edge.vertices[0]), get_depths(c1, edge.vertices[0])
            l01, l11 = get_depths(c0, edge.vertices[1]), get_depths(c1, edge.vertices[1])
            row_len = numQuadFaces - 1 if numQuadFaces > 1 else 1
            for (la0, la1) in [(l00, l10), (l01, l11)]:
                r, b = [0] * row_len, [0]
                if f0Idx == 0:
                    b[0] = la0
                    if f1Idx > 0: r[f1Idx - 1] = la1
                elif f1Idx == 0:
                    b[0] = la1
                    if f0Idx > 0: r[f0Idx - 1] = la0
                else:
                    r[f0Idx - 1] = la0
                    r[f1Idx - 1] = -la1
                matrixRows.append(r); rhRows.append(b)
        factors = [1.0]
        if numQuadFaces > 2 and len(matrixRows) > 0:
            m, b = Mat(matrixRows), Mat(rhRows)
            try: factors = [1] + [f[0] for f in m.solve(b)]
            except: factors = [1.0] * numQuadFaces
        elif numQuadFaces == 2 and len(matrixRows) > 0:
            if abs(matrixRows[0][0]) > 1e-6:
                factors = [1, 0.5 * (rhRows[0][0]/matrixRows[0][0] + rhRows[1][0]/matrixRows[1][0])]
            else: factors = [1, 1]
        final_quads = []
        for i, face in enumerate(computedCoordsByFace.keys()):
            quad = computedCoordsByFace[face]
            scale = factors[i] if i < len(factors) else 1.0
            final_quads.append([ [x * scale for x in q[:3]] for q in quad])
        name = inputMesh.name + '_3D'
        me = bpy.data.meshes.new(name)    
        ob = bpy.data.objects.new(name, me)    
        bpy.context.collection.objects.link(ob)    
        verts, faces, idx = [], [], 0
        for quad in final_quads:
            faces.append([idx, idx+1, idx+2, idx+3])
            verts.extend(quad); idx += 4
        me.from_pydata(verts, [], faces)    
        me.update(calc_edges=True)
        bpy.ops.object.select_all(action='DESELECT')
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        bpy.ops.object.mode_set(mode='OBJECT')
        return ob
    
    def getOutputMeshScale(self, camera, inMesh, outMesh):
        inMeanPos = [0.0] * 3
        cmi = camera.matrix_world.inverted()
        mm = inMesh.matrix_world
        for v in inMesh.data.vertices:
            vCamSpace =  cmi @ mm @ v.co.to_4d()
            for i in range(3): inMeanPos[i] += vCamSpace[i] / len(inMesh.data.vertices)
        outMeanPos = [0.0] * 3
        for v in outMesh.data.vertices:
            for i in range(3): outMeanPos[i] += v.co[i] / len(outMesh.data.vertices)
        inD = math.sqrt(sum([x*x for x in inMeanPos]))
        outD = math.sqrt(sum([x*x for x in outMeanPos]))
        return inD / outD if outD != 0 else 1
    
    def execute(self, context):
        self.camera = bpy.context.scene.camera
        self.mesh = bpy.context.active_object
        if not self.camera or not self.mesh or self.mesh.type != 'MESH':
            self.report({'ERROR'}, "Req: Active Mesh and Camera"); return {'CANCELLED'}
        computedCoordsByFace, quads = {}, []
        for f in self.mesh.data.polygons:
            if len(f.vertices) == 4:
                inputPointsCam = self.worldToCameraSpace([self.mesh.data.vertices[i] for i in f.vertices])
                qHats = [normalize(x) for x in inputPointsCam]
                outputPointsCam = self.computeQuadDepthInformation(*qHats)
                for i in range(4): outputPointsCam[i] = list(outputPointsCam[i]) + [f.vertices[i]]
                computedCoordsByFace[f] = outputPointsCam
                quads.append(outputPointsCam)
        m = self.createMesh(self.mesh, computedCoordsByFace, quads)
        m.matrix_world = self.camera.matrix_world
        m.scale = [self.getOutputMeshScale(self.camera, self.mesh, m)] * 3
        return{'FINISHED'}

class ApplyProjectedTexturesOperator(bpy.types.Operator):
    bl_idname = "object.apply_projected_textures"
    bl_label = "3. Apply Textures"
    bl_description = "Applies Subsurf and UV Project modifiers to bake the texture."

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh")
            return {'CANCELLED'}
        if context.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
        subsurf_mods = [m.name for m in obj.modifiers if m.type == 'SUBSURF']
        uv_mods = [m.name for m in obj.modifiers if m.type == 'UV_PROJECT']
        for name in subsurf_mods: bpy.ops.object.modifier_apply(modifier=name)
        for name in uv_mods: bpy.ops.object.modifier_apply(modifier=name)
        self.report({'INFO'}, "Textures Applied!")
        return {'FINISHED'}

class PhotoModelingToolsPanel(bpy.types.Panel):    
    bl_idname = "VIEW3D_PT_photo_modeling_tools"
    bl_label = "Photo Modeling Tools"    
    bl_space_type = "VIEW_3D"    
    bl_region_type = "UI"
    bl_category = "Photo Modeling Tools"

    def draw(self, context):
        l = self.layout
        scn = context.scene
        
        l.operator("object.load_image_setup_plane", icon='FILE_IMAGE')
        
        box = l.box()
        box.label(text="Alignment Axis:")
        box.prop(scn, "blam_axis_preset", text="")
        
        row = box.row()
        row.enabled = bool(context.mode == 'EDIT_MESH' or (context.active_object and context.active_object.type == 'MESH'))
        row.operator("object.solve_camera_from_mesh", icon='OUTLINER_OB_CAMERA')
        
        l.operator("object.apply_projected_textures", icon='TEXTURE')

classes = (
    LoadImageAndSetupPlaneOperator,
    SolveCameraFromMeshOperator,
    ProjectBackgroundImageOntoMeshOperator,
    Reconstruct3DMeshOperator,
    ApplyProjectedTexturesOperator,
    PhotoModelingToolsPanel,
)

def register():
    for cls in classes: bpy.utils.register_class(cls)
    
    # IMPROVEMENT: PROPERTY FOR AXIS DROPDOWN
    bpy.types.Scene.blam_axis_preset = bpy.props.EnumProperty(
        name="Axis Preset",
        description="Choose what world axis the active face represents",
        items=[
            ('XY', "Floor (XY)", "Align face to the world ground plane"),
            ('XZ', "Front Wall (XZ)", "Align face to the front wall plane"),
            ('YZ', "Side Wall (YZ)", "Align face to the side wall plane"),
        ],
        default='XY'
    )
 
def unregister():
    for cls in classes: bpy.utils.unregister_class(cls)
    del bpy.types.Scene.blam_axis_preset
 
if __name__ == "__main__":
    register()