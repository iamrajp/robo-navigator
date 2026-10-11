import math
import time
import signal
import sys
import numpy as np
import open3d as o3d
from pyrplidar import PyRPlidar

PORT       = '/dev/ttyUSB0'
BAUDRATE   = 460800
RANGE_M    = 6      # ignore points farther than this (metres) — tighter
MIN_POINTS = 80     # minimum valid points per scan — stricter
VOXEL_SIZE = 0.05   # downsample map at 5 cm resolution for faster ICP
FITNESS_OK = 0.5    # minimum ICP fitness to accept a scan — stricter
MAP_FILE   = 'home_map.pcd'

# ── helpers ───────────────────────────────────────────────────────────────────

def scan_to_pointcloud(points):
    """Convert [(angle_deg, distance_mm), ...] to Open3D PointCloud."""
    xyz = []
    for angle, distance in points:
        if distance <= 0:
            continue
        d_m = distance / 1000
        if d_m > RANGE_M:
            continue
        x = d_m * math.cos(math.radians(angle))
        y = d_m * math.sin(math.radians(angle))
        xyz.append([x, y, 0.0])

    pcd = o3d.geometry.PointCloud()
    if xyz:
        pcd.points = o3d.utility.Vector3dVector(np.array(xyz, dtype=np.float64))
    return pcd


def preprocess(pcd):
    """Downsample and remove outliers from a single scan before using it."""
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
    return pcd


def align_to_map(scan_pcd, map_pcd):
    """Align scan directly to the full accumulated map (frame-to-map ICP)."""
    map_down = map_pcd.voxel_down_sample(VOXEL_SIZE)

    result = o3d.pipelines.registration.registration_icp(
        scan_pcd,
        map_down,
        max_correspondence_distance=0.15,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    return result.transformation, result.fitness


def collect_scan(scan_generator):
    """Collect one full 360° sweep by detecting angle wraparound."""
    points   = []
    last_ang = None

    for scan in scan_generator():
        if scan.distance <= 0:
            continue
        angle = scan.angle

        if last_ang is not None and angle < 10 and last_ang > 350:
            if len(points) >= MIN_POINTS:
                return points
            points = []

        points.append((angle, scan.distance))
        last_ang = angle


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    lidar = PyRPlidar()
    lidar.connect(port=PORT, baudrate=BAUDRATE, timeout=3)
    lidar.stop()
    time.sleep(0.5)

    info   = lidar.get_info()
    health = lidar.get_health()
    print(f"Connected — model={info.model}  firmware={info.firmware_major}.{info.firmware_minor}")
    print(f"Health: status={health.status}  error_code={health.error_code}")

    lidar.set_motor_pwm(660)
    time.sleep(2)
    print("\nMotor running. Drive the robot slowly around every room.")
    print("Press Ctrl+C when done — map will be saved.\n")

    scan_generator = lidar.start_scan()

    # live visualiser
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Home Map — Live', width=800, height=700)

    map_pcd      = o3d.geometry.PointCloud()
    robot_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    robot_sphere.paint_uniform_color([1, 0, 0])
    vis.add_geometry(map_pcd)
    vis.add_geometry(robot_sphere)

    scan_count    = 0
    skipped_count = 0

    def shutdown(sig=None, frame=None):
        print(f"\n\nStopping — {scan_count} scans collected, {skipped_count} skipped.")
        lidar.stop()
        lidar.set_motor_pwm(0)
        lidar.disconnect()
        print(f"Saving map to {MAP_FILE} ...")
        o3d.io.write_point_cloud(MAP_FILE, map_pcd)
        print(f"Saved — {len(map_pcd.points)} total points.")
        vis.destroy_window()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    while True:
        raw = collect_scan(scan_generator)
        if not raw:
            continue

        current_pcd = preprocess(scan_to_pointcloud(raw))
        scan_count += 1

        if len(current_pcd.points) < MIN_POINTS:
            continue

        # ── first scan: place at origin ──
        if len(map_pcd.points) == 0:
            map_pcd += current_pcd
            vis.update_geometry(map_pcd)
            vis.poll_events()
            vis.update_renderer()
            print(f"Scan {scan_count:3d} | {len(current_pcd.points):4d} pts | origin")
            continue

        # ── align scan directly to the full map (no drift compounding) ──
        transform, fitness = align_to_map(current_pcd, map_pcd)

        if fitness < FITNESS_OK:
            skipped_count += 1
            print(f"Scan {scan_count:3d} | {len(current_pcd.points):4d} pts | "
                  f"poor fit={fitness:.2f} — skipped (move slower)")
            continue

        # move scan into map frame and merge
        current_pcd.transform(transform)
        map_pcd += current_pcd

        # update robot marker position
        robot_pos = transform[:3, 3]
        robot_sphere.translate(robot_pos - robot_sphere.get_center())

        vis.update_geometry(map_pcd)
        vis.update_geometry(robot_sphere)
        vis.poll_events()
        vis.update_renderer()

        print(f"Scan {scan_count:3d} | {len(current_pcd.points):4d} pts | "
              f"fit={fitness:.2f} | "
              f"pos=({robot_pos[0]:.2f}, {robot_pos[1]:.2f}) m")


if __name__ == '__main__':
    main()
