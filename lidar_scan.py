from rplidar import RPLidar
import time

PORT = '/dev/ttyUSB0'
BAUDRATE = 460800  # C1 uses 460800 (not 115200)

def main():
    lidar = RPLidar(PORT, baudrate=BAUDRATE)

    try:
        info = lidar.get_info()
        print(f"Connected: {info}")
        print(f"Health:    {lidar.get_health()}")

        lidar.start_motor()
        time.sleep(1)  # let motor spin up

        print("\nScanning... (Ctrl+C to stop)\n")
        for i, scan in enumerate(lidar.iter_scans()):
            print(f"--- Scan #{i+1} ({len(scan)} points) ---")
            for quality, angle, distance in scan:
                if quality > 0 and distance > 0:
                    print(f"  {angle:6.1f}°  {distance:7.1f} mm  (quality={quality})")
            if i >= 4:  # stop after 5 scans for demo
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        print("Lidar disconnected.")

if __name__ == '__main__':
    main()
