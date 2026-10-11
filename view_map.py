import open3d as o3d
import sys

map_file = sys.argv[1] if len(sys.argv) > 1 else 'home_map.pcd'

print(f"Loading {map_file} ...")
pcd = o3d.io.read_point_cloud(map_file)
print(f"  {len(pcd.points)} points loaded")

# colour points cyan
pcd.paint_uniform_color([0, 1, 1])

# add a red sphere at origin (robot start position)
robot = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
robot.paint_uniform_color([1, 0, 0])
robot.translate([0, 0, 0])

# coordinate frame at origin
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

print("\nControls:")
print("  Left click + drag  → rotate")
print("  Right click + drag → pan")
print("  Scroll wheel       → zoom")
print("  R                  → reset view")
print("  Q / Esc            → quit\n")

o3d.visualization.draw_geometries(
    [pcd, robot, frame],
    window_name='Home Map',
    width=900,
    height=700,
    zoom=0.5
)
