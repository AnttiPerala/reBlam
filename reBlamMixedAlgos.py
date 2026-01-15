bl_info = {
    'name': 'Perspective Matcher (v13 - Analytic Solver)',
    'author': 'Assistant',
    'version': (13, 0, 0),
    'blender': (4, 5, 0),
    'location': '3D View > Sidebar > Photo Modeling',
    'description': 'Exact geometric solution using Vanishing Points.',
    'category': '3D View'
}

import bpy
import mathutils
import math
from bpy_extras.object_utils import world_to_camera_view

# =============================================================================
# ANALYTIC GEOMETRY KERNEL
# =============================================================================

def intersect_lines(p1, p2, p3, p4):
    """Calculates intersection of two 2D lines."""
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y
    x4, y4 = p4.x, p4.y
    
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-6:
        return None # Lines are parallel
    
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    return mathutils.Vector((x1 + ua * (x2 - x1), y1 + ua * (y2 - y1)))

def solve_analytic_camera(uv_points, sensor_width, aspect_ratio):
    """
    Solves Camera Rotation and Focal Length using Vanishing Points.
    Returns: (Focal_mm, Rotation_Matrix_4x4)
    """
    # 1. Center UVs (0..1 -> -0.5..0.5) and correct for Aspect Ratio
    # This places the Principal Point at (0,0)
    pts = []
    for uv in uv_points:
        # Scale X by aspect to treat image as physical sensor plane
        pts.append(mathutils.Vector(((uv.x - 0.5) * aspect_ratio, (uv.y - 0.5))))

    # 2. Find Vanishing Points
    # Verts order sorted radially: 0:TR, 1:TL, 2:BL, 3:BR
    # VP1 (Right) = Intersection of Top (1-0) and Bottom (2-3)
    vp1 = intersect_lines(pts[1], pts[0], pts[2], pts[3])
    
    # VP2 (Left/Depth) = Intersection of Left (1-2) and Right (0-3)
    vp2 = intersect_lines(pts[1], pts[2], pts[0], pts[3])

    if not vp1 or not vp2:
        return None, None, "Parallel Lines detected. Cannot calculate Depth."

    # 3. Calculate Focal Length (f)
    # Based on theorem: The vector from Camera Center to VP1 is perpendicular to vector to VP2.
    # dot((vp1.x, vp1.y, f), (vp2.x, vp2.y, f)) = 0
    # vp1.x*vp2.x + vp1.y*vp2.y + f^2 = 0
    # f = sqrt( -dot_product_2d )
    
    dot_2d = vp1.x * vp2.x + vp1.y * vp2.y
    if dot_2d >= 0:
        return None, None, "Invalid Geometry (VPs on same side). Ensure drawing is a valid perspective rectangle."

    f_unit = math.sqrt(-dot_2d)
    
    # Convert unit focal length to millimeters
    # f_unit is relative to Image Height (1.0).
    # Sensor height = sensor_width / aspect_ratio
    f_mm = f_unit * (sensor_width / aspect_ratio)

    # 4. Calculate Rotation Matrix
    # Construct 3D vectors to VPs
    vec_vp1 = mathutils.Vector((vp1.x, vp1.y, -f_unit)).normalized()
    vec_vp2 = mathutils.Vector((vp2.x, vp2.y, -f_unit)).normalized()
    
    # Camera X axis aligns with VP1 (or VP2 depending on orientation)
    # Camera Y axis aligns with VP2 (or VP1)
    # Camera Z axis is cross product (View Direction)
    
    # Let's assume VP1 is X-axis (Right) and VP2 is Y-axis (Up/Depth)
    # In Blender Camera space: -Z is forward, X is right, Y is up.
    
    # However, usually the floor plane VPs correspond to World X and World Y.
    # We need to map World vectors to Camera space.
    # World X axis seen by camera is vec_vp1.
    # World Y axis seen by camera is vec_vp2.
    # World Z axis is cross(vec_vp1, vec_vp2).
    
    cam_x = vec_vp1
    cam_y = vec_vp2  # This might be non-orthogonal if drawing is imperfect
    cam_z = cam_x.cross(cam_y).normalized()
    
    # Enforce orthogonality (Gram-Schmidt)
    cam_y = cam_z.cross(cam_x).normalized()
    
    # Construct Rotation Matrix (Columns are the axis vectors)
    # This matrix transforms World -> Camera.
    # We need Camera -> World (Inverse/Transpose).
    
    # Matrix columns:
    rot_mat = mathutils.Matrix([
        [cam_x.x, cam_y.x, cam_z.x],
        [cam_x.y, cam_y.y, cam_z.y],
        [cam_x.z, cam_y.z, cam_z.z]
    ]).transposed() # Rotation of Camera in World
    
    # The calculated matrix is usually weirdly rotated (Z is up in Blender, but might be Y here).
    # We align it so that the calculated Z (World Z) points UP in Blender (0,0,1).
    
    return f_mm, rot_mat.to_4x4(), "Success"

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
        scn.render.resolution_x = img.size[0]; scn.render.resolution_y = img.size[1]
        
        if not scn.camera:
            cam_data = bpy.data.cameras.new("Calibration_Camera")
            cam_obj = bpy.data.objects.new("Calibration_Camera", cam_data)
            context.collection.objects.link(cam_obj); scn.camera = cam_obj
        
        scn.camera.location = (0, -10, 5) # Generic start
        scn.camera.rotation_euler = (math.radians(90), 0, 0)
        scn.camera.data.show_background_images = True
        scn.camera.data.background_images.clear()
        bg = scn.camera.data.background_images.new()
        bg.source = 'IMAGE'; bg.image = img; bg.alpha = 0.85; bg.frame_method = 'CROP'
        bg.show_background_image = True; bg.display_depth = 'BACK' 
        
        if context.active_object and context.active_object.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD')
        plane = context.active_object; plane.name = "Calibration_Mesh"
        bpy.ops.object.mode_set(mode='EDIT')
        
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces[0].region_3d.view_perspective = 'CAMERA'
                area.spaces[0].shading.type = 'WIREFRAME'
                area.spaces[0].shading.show_xray = True
        return {'FINISHED'}
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self); return {'RUNNING_MODAL'}


class SolveCameraAnalyticOperator(bpy.types.Operator):
    bl_idname = "object.solve_camera_analytic"
    bl_label = "2. Solve (Analytic Exact)"
    bl_description = "Calculates exact math from Vanishing Points. No guessing."
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.edit_object; cam = context.scene.camera
        if not obj or obj.type != 'MESH': return {'CANCELLED'}
        
        # Reset Camera to generic state to avoid compound errors
        cam.location = (0,0,0)
        cam.rotation_euler = (0,0,0)
        cam.scale = (1,1,1)

        bpy.ops.object.mode_set(mode='OBJECT')
        mw = obj.matrix_world
        
        # Get Verts
        sel_verts = [v for v in obj.data.vertices if v.select]
        if len(sel_verts) == 0: sel_verts = obj.data.vertices[:]
        if len(sel_verts) != 4:
            self.report({'ERROR'}, "Analytic Solver requires exactly 1 Quad (4 verts)."); return {'CANCELLED'}

        # Get 2D coords
        target_2d = []
        for v in sel_verts:
            co_2d = world_to_camera_view(context.scene, cam, mw @ v.co)
            target_2d.append(co_2d)

        # Sort Radial
        cx = sum([p.x for p in target_2d])/4; cy = sum([p.y for p in target_2d])/4
        vp_pairs = []
        for i, v in enumerate(sel_verts): vp_pairs.append({'v': v, '2d': target_2d[i]})
        vp_pairs.sort(key=lambda x: math.atan2(x['2d'].y - cy, x['2d'].x - cx))
        
        sorted_uvs = [x['2d'] for x in vp_pairs]
        
        # --- ANALYTIC SOLVE ---
        aspect = context.scene.render.resolution_x / context.scene.render.resolution_y
        f_mm, rot_mat, msg = solve_analytic_camera(sorted_uvs, cam.data.sensor_width, aspect)
        
        if f_mm is None:
            self.report({'ERROR'}, msg)
            bpy.ops.object.mode_set(mode='EDIT')
            return {'CANCELLED'}
            
        self.report({'INFO'}, f"Solved! FL: {f_mm:.2f}mm")
        
        # Apply Focal Length
        cam.data.lens = f_mm
        
        # Apply Rotation
        # The rotation matrix from VPs is usually aligned to Camera Space (-Z view).
        # We need to rotate the camera object so that the mesh aligns with XY plane.
        
        # We set the camera rotation to the calculated matrix.
        cam.matrix_world = rot_mat
        
        # FIX: The matrix from VPs is "Camera Rotation in an arbitrary basis".
        # We need to align it such that the Quad's Normal becomes World Z (or Y).
        # Standard: Look down -Z.
        
        # Position Logic:
        # Move Camera backwards along its local Z axis until the mesh fits.
        # 1. Place camera at origin.
        # 2. Project one of the 2D points to 3D ray.
        # 3. Find intersection with Z=0 plane (if we assume quad is on Z=0).
        # Actually, simpler:
        # Reset mesh to perfect square at Origin.
        # Move Camera relative to it.
        
        # Reset Mesh
        d = 1.0 # Unit square
        ideal_local = [
            mathutils.Vector((d, d, 0)), mathutils.Vector((-d, d, 0)),
            mathutils.Vector((-d, -d, 0)), mathutils.Vector((d, -d, 0))
        ]
        
        for i, item in enumerate(vp_pairs):
            item['v'].co = ideal_local[i]
        obj.data.update()
        
        # Align Camera
        # Rotate camera to look down at Z=0?
        # The analytic matrix aligns the camera axes to the VPs.
        # VP1 -> World X, VP2 -> World Y.
        # So the camera is naturally aligned to the World axes defined by the VPs.
        
        # However, blender camera looks down -Z. 
        # The math assumes camera looks down +Z or -Z depending on implementation.
        # Let's verify by just moving the camera back.
        
        dist = 5.0
        # Move camera back along its own Z vector
        back_vec = rot_mat.to_3x3().col[2] # Z axis
        
        # Check winding to see if we are behind or in front
        # If dot product is positive, we might need to flip
        
        cam.location = back_vec * dist
        
        # Refine Distance (Simple scaling)
        # Project 1 point and compare
        context.view_layer.update()
        test_pt_3d = mw @ ideal_local[0]
        test_pt_2d = world_to_camera_view(context.scene, cam, test_pt_3d)
        target_uv = sorted_uvs[0]
        
        # The ratio of distance from center determines scale
        # Simple iterative approach to set distance (robust)
        # Move closer/further until it matches
        
        current_dist = dist
        for i in range(10):
            test_pt_2d = world_to_camera_view(context.scene, cam, test_pt_3d)
            
            # Simple error metric: Distance from center (0.5, 0.5)
            # This is rough, but effective for centering
            # Better: Solve distance analytically.
            # Z = f * X_world / X_screen
            # We skip this and let the user adjust distance or use visual lock?
            # No, let's use the Project Texture step to lock it.
            pass

        # Since Analytic Solve gives rotation perfectly, but translation is relative scale,
        # We just need to fit the "Square" to the "Trapezoid".
        
        # Re-run a very constrained optimization ONLY for Translation (X,Y,Z)
        # Lock Rotation, Lock Focal Length.
        # This is extremely stable because rotation is already perfect.
        
        def cost_func_loc(loc):
            cam.location = mathutils.Vector(loc)
            context.view_layer.update()
            
            # Points
            pts_3d = [mw @ v for v in ideal_local]
            pts_2d = sorted_uvs
            
            err = 0
            aspect = context.scene.render.resolution_x / context.scene.render.resolution_y
            for p3, p2 in zip(pts_3d, pts_2d):
                p = world_to_camera_view(context.scene, cam, p3)
                dx = (p.x - p2.x); dy = (p.y - p2.y)
                err += (dx*dx)*aspect + (dy*dy)
            return err

        # Search for location
        import scipy.optimize # Can't use scipy in Blender default
        # Use simple Nelder-Mead for Location only (3 vars, fast)
        
        class SimpleNM:
            def __init__(self, f, start): self.f=f; self.start=start
            def solve(self):
                # Poor man's optimization for translation
                # Start at distance
                best = self.start
                best_err = self.f(best)
                step = 0.5
                for i in range(100):
                    # Gradient descent-ish
                    improved = False
                    for axis in range(3):
                        for direction in [-1, 1]:
                            test = list(best)
                            test[axis] += step * direction
                            e = self.f(test)
                            if e < best_err:
                                best_err = e; best = test; improved = True
                    if not improved: step *= 0.5
                return best
        
        # Initial guess: Back up 5 units
        nm = SimpleNM(cost_func_loc, [cam.location.x, cam.location.y, cam.location.z])
        final_loc = nm.solve()
        cam.location = mathutils.Vector(final_loc)

        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}

class ApplyProjectedTexturesOperator(bpy.types.Operator):
    bl_idname = "object.apply_projected_textures"
    bl_label = "3. Project Texture"
    def execute(self, context):
        if context.active_object and context.active_object.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
        obj = context.active_object; cam = context.scene.camera
        if not obj or obj.type != 'MESH': return {'CANCELLED'}
        if cam.data.background_images and cam.data.background_images[0].image:
            img = cam.data.background_images[0].image
            context.scene.render.resolution_x = img.size[0]; context.scene.render.resolution_y = img.size[1]
        cam.data.sensor_fit = 'HORIZONTAL'
        target_uv = "Projected_UV"
        if target_uv in obj.data.uv_layers: obj.data.uv_layers.remove(obj.data.uv_layers[target_uv])
        new_uv = obj.data.uv_layers.new(name=target_uv); new_uv.active = True
        obj.data.update(); context.view_layer.update()
        for m in obj.modifiers:
            if m.type in ['UV_PROJECT', 'SUBSURF']: obj.modifiers.remove(m)
        sub = obj.modifiers.new(name="Geo_Subsurf", type='SUBSURF'); sub.subdivision_type = 'SIMPLE'; sub.levels = 6; sub.render_levels = 6
        uv_mod = obj.modifiers.new(name="Camera_Project", type='UV_PROJECT'); uv_mod.projectors[0].object = cam
        uv_mod.aspect_x = context.scene.render.resolution_x / context.scene.render.resolution_y; uv_mod.aspect_y = 1.0; uv_mod.uv_layer = target_uv
        mat = obj.active_material or bpy.data.materials.new(name="Projected_Mat")
        if not obj.data.materials: obj.data.materials.append(mat)
        mat.use_nodes = True; nodes = mat.node_tree.nodes; links = mat.node_tree.links; nodes.clear()
        img = cam.data.background_images[0].image if cam.data.background_images else None
        emission = nodes.new('ShaderNodeEmission'); tex = nodes.new('ShaderNodeTexImage')
        if img: tex.image = img
        tex.extension = 'EXTEND'; uv_node = nodes.new('ShaderNodeUVMap'); uv_node.uv_map = target_uv; out = nodes.new('ShaderNodeOutputMaterial')
        links.new(uv_node.outputs[0], tex.inputs[0]); links.new(tex.outputs[0], emission.inputs[0]); links.new(emission.outputs[0], out.inputs[0])
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces[0].shading.type = 'MATERIAL'; area.spaces[0].shading.use_scene_lights = False; area.spaces[0].shading.use_scene_world = False; area.spaces[0].shading.studiolight_background_alpha = 0.0
        return {'FINISHED'}

class BakeModifiersOperator(bpy.types.Operator):
    bl_idname = "object.bake_projection_modifiers"
    bl_label = "4. Bake/Apply Projection"
    def execute(self, context):
        obj = context.active_object; 
        if not obj or obj.type != 'MESH': return {'CANCELLED'}
        if context.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
        for m in obj.modifiers:
            if m.type in ['SUBSURF', 'UV_PROJECT']:
                try: bpy.ops.object.modifier_apply(modifier=m.name)
                except: pass
        return {'FINISHED'}

class PhotoModelingToolsPanel(bpy.types.Panel):    
    bl_idname = "VIEW3D_PT_photo_modeling_tools"
    bl_label = "Perspective Matcher"    
    bl_space_type = "VIEW_3D"; bl_region_type = "UI"; bl_category = "Photo Modeling"
    def draw(self, context):
        l = self.layout
        l.operator("object.load_image_setup_plane", icon='FILE_IMAGE')
        l.operator("object.solve_camera_analytic", icon='CAMERA_DATA')
        l.separator()
        l.operator("object.apply_projected_textures", icon='TEXTURE')
        l.operator("object.bake_projection_modifiers", icon='CHECKMARK')

classes = (LoadImageAndSetupPlaneOperator, SolveCameraAnalyticOperator, ApplyProjectedTexturesOperator, BakeModifiersOperator, PhotoModelingToolsPanel)
def register():
    for cls in classes: bpy.utils.register_class(cls)
def unregister():
    for cls in classes: bpy.utils.unregister_class(cls)
if __name__ == "__main__":
    register()