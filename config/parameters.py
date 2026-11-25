import os

# Project root = folder that contains config/. Relative paths below start from here,
# so the scripts work no matter from which folder they are started.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_path(path):
    """Relative path -> path inside the project; absolute paths are kept."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)

class ModelParameters:
    def __init__(self):
        self.img_size = (1600,1600)
        self.pixels_per_meter = 16 
        self.colors = {'Car':(0,0,0), 'Van':(255,0,0),'Pedestrian':(125,125,0),'Cyclist':(255,255,0),'Truck':(255,0,255),'Motorcycle':(0,255,0),'Bus':(0,0,255)}
        self.zones = [(800,550,1050,750),(650,550,800,700),(500,550,650,780),(450,350,650,500),(550,780,750,900)]
        # zones used by collect_data.py (data collection)
        self.collect_zones = [(920,450,1050,750),(650,550,800,700),(500,550,650,780),(450,300,700,500)]
        self.delta_t = 1                 # prediction horizon [s]
        # Monte-Carlo samples per object pair and zone
        self.mc_samples_visualization = 50
        self.mc_samples_collect = 100
        # visualization.py stops after this many frames (None = run through all data)
        self.max_frames = None
        self.risk = {'LOW':(0,255,0),'MEDIUM':(255,255,0),'HIGH':(255,170,100),'CRITICAL':(255,0,0)}
        

        # ---------------- motion model per object class ----------------
        # Heading: above motion_min_speed the direction of the velocity vector is
        # used instead of the bounding-box yaw (the box yaw of pedestrians is ~90deg
        # off their walking direction, and box yaw can flip by 180deg).
        self.motion_min_speed = 0.5      # m/s
        # Angular velocity: slope of the velocity heading over this time window.
        self.omega_window_sec = 0.8      # s
        # Per class:
        #   model       : 'CTRV' or 'CV' (constant velocity, omega forced to 0)
        #   max_omega   : |omega| is clipped to this value [rad/s]
        #   sigma_pos/speed/yaw/omega : initial state uncertainty
        #   sigma_a     : (min, max) clip for the estimated acceleration noise [m/s^2]
        #   sigma_alpha : (min, max) clip for the estimated yaw-acceleration noise [rad/s^2]
        vehicle = dict(model='CTRV', max_omega=1.0,
                       sigma_pos=0.5, sigma_speed=0.5, sigma_yaw=0.05, sigma_omega=0.10,
                       sigma_a=(0.5, 3.0), sigma_alpha=(0.1, 1.0))
        heavy = dict(vehicle, max_omega=0.6, sigma_a=(0.3, 2.0), sigma_alpha=(0.05, 0.6))
        self.class_motion = {
            'Car': vehicle,
            'Van': vehicle,
            'Motorcycle': dict(vehicle, sigma_yaw=0.08, sigma_omega=0.15),
            'Truck': heavy,
            'Bus': heavy,
            'Cyclist': dict(model='CTRV', max_omega=1.0,
                            sigma_pos=0.3, sigma_speed=0.4, sigma_yaw=0.15, sigma_omega=0.30,
                            sigma_a=(0.3, 2.0), sigma_alpha=(0.3, 1.5)),
            'Pedestrian': dict(model='CV', max_omega=0.0,
                               sigma_pos=0.3, sigma_speed=0.3, sigma_yaw=0.35, sigma_omega=0.50,
                               sigma_a=(0.2, 1.0), sigma_alpha=(0.5, 2.0)),
            'default': vehicle,
        }


class DataConfig:
    """Data locations, relative to the project root (absolute paths also work).

    Change them here, or with environment variables, e.g.
        LIDAR_DATA_PATH=data/2026-05-21 python collect_data.py
    """
    def __init__(self):
        # folder with the object-list JSON files (one frame per file or 5 minutes per file)
        self.data_path = project_path(os.environ.get('LIDAR_DATA_PATH', 'data'))
        # folder with the traffic-light CSV files (not used yet)
        self.trafficlight_csv_dir = project_path(os.environ.get('LIDAR_TRAFFICLIGHT_PATH', 'data/trafficlight'))
        # output file of collect_data.py
        self.output_csv = project_path(os.environ.get('LIDAR_OUTPUT_CSV', 'data.csv'))
        # background map for visualization.py
        self.background_image = project_path('sample/305_rev.png')
