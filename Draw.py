#-------------------------------------------------------------------------------
#
# Draw.py -- new flock experiments
#
# Graphics utilities based on Open3D
#
# MIT License -- Copyright © 2023 Craig Reynolds
#
#-------------------------------------------------------------------------------

import open3d as o3d
import numpy as np # temp?
import time
import Utilities as util  # temp?
from Vec3 import Vec3     # temp?


class Draw:
    """Graphics utilities based on Open3D."""

    # Initialize new instance.
    def __init__(self):
        self.temp = 0

    # Class properties to hold raw scene data for Open3D TriangleMesh.
#    mesh_vertex_count = 0
    mesh_vertices = o3d.utility.Vector3dVector()
    mesh_triangles = o3d.utility.Vector3iVector()
    mesh_vertex_colors = o3d.utility.Vector3dVector()
    
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    # TODO 20230426 add line drawing support for annotation
#    line_vertex_count = 0
    line_points = o3d.utility.Vector3dVector()
    line_segments = o3d.utility.Vector2iVector()
    line_colors = o3d.utility.Vector3dVector()
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    # class storage of current visualizer
    vis = None
    frame_start_time = None
    frame_duration = 0.01
    
    @staticmethod
    def add_triangle_single_color(v1, v2, v3, color):
        color = color.asarray()
        for v in [v1, v2, v3]:
            Draw.mesh_vertices.append(v.asarray())
            Draw.mesh_vertex_colors.append(color)
        t = len(Draw.mesh_triangles) * 3
        Draw.mesh_triangles.append([t, t + 1, t + 2])

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    # TODO 20230426 add line drawing support for annotation
    @staticmethod
    def add_line_segment(v1, v2, color1=None, color2=None):
        if color1 == None:
#            color1 = Vec3(0.5, 0.5, 0.5)
            color1 = Vec3(1, 0, 0)
        if color2 == None:
            color2 = color1
        Draw.line_points.append(v1.asarray())
        Draw.line_points.append(v2.asarray())
        Draw.line_colors.append(color1.asarray())
        Draw.line_colors.append(color2.asarray())
        i = len(Draw.line_segments) * 2
        Draw.line_segments.append([i, i + 1])
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    @staticmethod
    def start_visualizer():
        # Create a mesh from the triangle vertices and indices
        Draw.triangle_mesh = o3d.geometry.TriangleMesh()
        Draw.triangle_mesh.vertices = Draw.mesh_vertices
        Draw.triangle_mesh.triangles = Draw.mesh_triangles
        Draw.triangle_mesh.vertex_colors = Draw.mesh_vertex_colors
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        # TODO 20230426 add line drawing support for annotation
        Draw.line_set = o3d.geometry.LineSet()
        Draw.line_set.points = Draw.line_points
        Draw.line_set.lines = Draw.line_segments
        Draw.line_set.colors = Draw.line_colors
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        # Create Visualizer add mesh, enter draw loop.
        Draw.vis = o3d.visualization.Visualizer()
        Draw.vis.create_window()
        Draw.vis.add_geometry(Draw.triangle_mesh)
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        # TODO 20230426 add line drawing support for annotation
        Draw.vis.add_geometry(Draw.line_set)
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        
        # TODO 23230411 temp ball for camera aim reference
        ball = o3d.geometry.TriangleMesh.create_sphere(0.1, 10)
        ball.compute_vertex_normals()
        ball.paint_uniform_color([0.8, 0.1, 0.1])
        Draw.vis.add_geometry(ball)

        Draw.frame_start_time = time.time()

        
    @staticmethod
    def close_visualizer():
        Draw.vis.destroy_window()

    @staticmethod
    def still_running():
        return Draw.vis.poll_events()

    @staticmethod
    def clear_scene():
#        Draw.mesh_vertex_count = 0
        Draw.mesh_vertices.clear()
        Draw.mesh_triangles.clear()
        Draw.mesh_vertex_colors.clear()
        
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        # TODO 20230426 add line drawing support for annotation
        Draw.line_points.clear()
        Draw.line_segments.clear()
        Draw.line_colors.clear()
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##


    @staticmethod
    def update_scene():
        Draw.triangle_mesh.vertices = Draw.mesh_vertices
        Draw.triangle_mesh.triangles = Draw.mesh_triangles
        Draw.vis.update_geometry(Draw.triangle_mesh)

        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        # TODO 20230426 add line drawing support for annotation
        Draw.line_set.points = Draw.line_points
        Draw.line_set.lines = Draw.line_segments
        Draw.line_set.colors = Draw.line_colors
        Draw.vis.update_geometry(Draw.line_set)
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

        Draw.vis.update_renderer()
        # Measure frame duration:
        frame_end_time = time.time()
        Draw.frame_duration = frame_end_time - Draw.frame_start_time
        Draw.frame_start_time = frame_end_time
        Draw.frame_counter += 1

    # TODO 20230419 but does not work because "Visualizer.get_view_control()
    #               gives a copy." https://github.com/isl-org/Open3D/issues/6009
    @staticmethod
    def update_camera(lookat):
        camera = Draw.vis.get_view_control()
        camera.set_lookat(lookat)

    # Frame counter
    frame_counter = 0
    
    
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ##
    ## TODO 20230419 random test code, to be removed eventually.

    # TODO 20230401
    # Trying to test per-triangle colors as claimed to exist on:
    # https://github.com/isl-org/Open3D/issues/1087#issuecomment-1222782733
    # See my questions on:
    # https://stackoverflow.com/questions/75907926/open3d-color-per-triangle
    # https://github.com/isl-org/Open3D/issues/6060
    def face_color_test():
        mesh = o3d.geometry.TriangleMesh.create_octahedron()
        rgb = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
        # face_colors = o3d.utility.Vector3dVector(np.array(rgb, np.float32))
        face_colors = o3d.utility.Vector3iVector(np.array(rgb, np.int32))
        # mesh.triangles["colors"] = face_colors
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh)
        vis.run()
        vis.destroy_window()

    # TODO 20230412, cannot set any view parameters because “get_view_control()
    # gives a copy” (https://github.com/isl-org/Open3D/issues/6009). So this
    # example (from https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/customized_visualization.py#L39):
    def custom_draw_geometry_with_rotation(pcd):
        def rotate_view(vis):
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
            return False
        o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                                  rotate_view)
    # does not work:
    def test_callback():
        mesh = o3d.geometry.TriangleMesh.create_octahedron()
        Draw.custom_draw_geometry_with_rotation(mesh)

    # Test animation callback.
    def test_animation_callback():
        mesh = o3d.geometry.TriangleMesh.create_octahedron()
        def recolor(vis):
            mesh.paint_uniform_color(np.random.rand(3))
            vis.update_geometry(mesh)
            return False
        o3d.visualization.draw_geometries_with_animation_callback([mesh], recolor)
#        o3d.visualization.draw([mesh], on_animation_frame=recolor)
#        o3d.visualization.draw([mesh], on_animation_tick=recolor)


    ##
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
