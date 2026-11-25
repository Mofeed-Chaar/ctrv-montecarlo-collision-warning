"""Collect collision alarms into a CSV file (no visualisation).

Same pipeline as visualization.py (heading from velocity, omega from heading
history, per-class CTRV/CV prediction, Monte-Carlo collision per zone), but every
non-SAFE pair is written to the CSV in the ORIGINAL sensor frame. The file is
flushed after every row, so it is complete up to the moment the program stops.

Run:  python collect_data.py
"""
import glob
import os
import csv
import atexit

import tqdm
import numpy as np
from config import parameters
from utils import get_json_file
from proccess import get_possition, get_shifted_coordinate, inside_image
from collision_predict import monte_carlo_all_pairs_collision_px_region, classify_risk
from trajectory import predict_gaussians_for_objects, wrap_angle


param = parameters.ModelParameters()
data = parameters.DataConfig()

zones = getattr(param, 'collect_zones', param.zones)
delta_t = param.delta_t
mc_samples = getattr(param, 'mc_samples_collect', 100)
non_motorized = ['Pedestrian', 'Cyclist']

# ---------------- incremental CSV output ----------------
# Every alarm row is written and flushed immediately, so the CSV is complete up
# to the moment the program stops (Ctrl+C, crash, ...), not only at the end.
csv_path = getattr(data, 'output_csv', 'data.csv')

# Per object (object_1_*, object_2_*). All values in the original sensor frame.
#   x, y            current position [m]
#   yaw             heading used by the model (velocity direction when moving) [rad]
#   box_yaw         bounding-box yaw reported by the sensor [rad]
#   speed, vx, vy   current speed / velocity components [m/s]
#   omega           measured turn rate [rad/s] (before the per-class clip; pedestrians are predicted with 0)
#   is_moving       speed >= motion_min_speed (heading taken from velocity)
#   width, height   box width / length [m]
#   mu_*, sigma_*   predicted position Gaussian after delta_t
#   sigma_a, sigma_alpha  process noise used for this object
#   track_frames    number of frames this object has been tracked (low = less reliable)
_vehicle_fields = ['type', 'id', 'x', 'y', 'yaw', 'speed', 'width', 'height',
				'mu_x', 'mu_y', 'sigma_xx', 'sigma_xy', 'sigma_yy',
				'vx', 'vy', 'box_yaw', 'omega', 'is_moving',
				'sigma_a', 'sigma_alpha', 'track_frames']
CSV_COLUMNS = ([f'object_1_{f}' for f in _vehicle_fields] +
			[f'object_2_{f}' for f in _vehicle_fields] +
			['collision_probability_global', 'collision_probability_given_region',
			'time_stamp', 'area_number',
			'risk_level', 'distance',
			'mean_continuous_risk', 'max_continuous_risk',
			'num_samples', 'num_region_samples', 'num_collision_samples',
			'nanosec', 'time_sec', 'delta_t', 'source_file'])

csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
atexit.register(csv_file.close)
csv_writer = csv.writer(csv_file)
csv_writer.writerow([''] + CSV_COLUMNS)   # first column = row index (as pandas index=True)
csv_file.flush()
csv_row_index = 0


def write_csv_row(row):
	"""Append one row to the CSV and push it to disk right away."""
	global csv_row_index
	csv_writer.writerow([csv_row_index] + [row.get(c, '') for c in CSV_COLUMNS])
	csv_row_index += 1
	csv_file.flush()
	os.fsync(csv_file.fileno())


print(f'Number of zones is {len(zones)}')

data_paths = sorted(glob.glob(os.path.join(data.data_path, '*.json')))
print(f'Number of files {len(data_paths)}')

loop = tqdm.tqdm(data_paths)
count = 0
objects = {}
frame_info = {}   # obj_id -> raw values of the current frame (sensor frame)

motion_min_speed = getattr(param, 'motion_min_speed', 0.5)
omega_window_sec = getattr(param, 'omega_window_sec', 0.8)
class_motion = getattr(param, 'class_motion', None)


def motion_heading(rotation_z, velocity_x, velocity_y, min_speed=motion_min_speed):
	"""Heading used by the motion model.

	Moving objects: direction of the velocity vector. The bounding-box yaw is not
	reliable for this (pedestrian boxes are ~90 deg off the walking direction and
	box yaw can flip by 180 deg). Standing objects: bounding-box yaw.
	Returns (heading, is_moving).
	"""
	if np.hypot(velocity_x, velocity_y) >= min_speed:
		return float(np.arctan2(velocity_y, velocity_x)), True
	return float(rotation_z), False


def compute_angular_velocity(current_heading, current_time_sec, is_moving, previous_state,
		window_sec=omega_window_sec):
	"""Angular velocity = slope of the (unwrapped) velocity heading over the last
	`window_sec` seconds (least-squares fit), using only frames where the object moves.

	The sensor field 'velocity_angular' is not used: it is derived from the box yaw
	and contains jumps of ~pi/0.1s = 31 rad/s when the box flips by 180 deg.
	previous_state: list of (time_sec, heading, is_moving) of earlier frames (or None).
	"""
	if not previous_state or not is_moving:
		return 0.0

	samples = [(t, h) for (t, h, moving) in previous_state
			if moving and current_time_sec - t <= window_sec and t < current_time_sec]
	samples.append((current_time_sec, current_heading))
	if len(samples) < 3:
		return 0.0

	t = np.array([s[0] for s in samples], dtype=float)
	h = np.unwrap(np.array([s[1] for s in samples], dtype=float))
	if t[-1] - t[0] < 0.25 * window_sec:
		return 0.0

	return float(np.polyfit(t - t[0], h, 1)[0])


heading_history = {}   # obj_id -> [(time_sec, heading, is_moving), ...]


def to_sensor_frame(vehicle, pred):
	"""Convert one result['vehicle_*'] entry + its prediction back to the original sensor frame.

	Internal (prediction) frame: x = position_y, y = position_x,
	yaw = -heading + pi/2. The inverse swaps x/y back, yaw -> pi/2 - yaw,
	and swaps the covariance axes (sigma_xx <-> sigma_yy, sigma_xy unchanged).
	"""
	x_int, y_int = vehicle['center_current']
	mu_int = pred['mu']
	S = pred['Sigma']
	heading = wrap_angle(np.pi/2 - vehicle['yaw'])
	return {
		'type': vehicle['type'],
		'id': vehicle['id'],
		'x': y_int,
		'y': x_int,
		'yaw': heading,
		'speed': vehicle['speed'],
		'width': vehicle['width'],
		'height': vehicle['height'],
		'mu_x': mu_int[1],
		'mu_y': mu_int[0],
		'sigma_xx': S[1][1],
		'sigma_xy': S[0][1],
		'sigma_yy': S[0][0],
	}


def normalize_object_detect(object_detect):
	"""Convert an object of the new 5-minute file format to the old field names.

	New format: track_id, class_name, bounding_box.{position, scale, yaw}, velocity.{x, y}
	Old format (returned unchanged): obj_id, obj_type, position_x, rotation_z, velocity_x, ...
	The old 'velocity_angular' is not needed: omega is computed from the heading history.
	"""
	if 'obj_id' in object_detect and 'obj_type' in object_detect:
		return object_detect

	bounding_box = object_detect.get('bounding_box', {})
	position = bounding_box.get('position', {})
	scale = bounding_box.get('scale', {})
	velocity = object_detect.get('velocity', {})
	velocity_x = float(velocity.get('x', 0.0))
	velocity_y = float(velocity.get('y', 0.0))

	return {
		'obj_id': object_detect.get('track_id'),
		'obj_type': object_detect.get('class_name'),
		'position_x': float(position.get('x', 0.0)),
		'position_y': float(position.get('y', 0.0)),
		'position_z': float(position.get('z', 0.0)),
		'scale_x': float(scale.get('x', 0.0)),
		'scale_y': float(scale.get('y', 0.0)),
		'scale_z': float(scale.get('z', 0.0)),
		'rotation_z': float(bounding_box.get('yaw', 0.0)),
		'velocity_x': velocity_x,
		'velocity_y': velocity_y,
		'velocity_magnitude': float(np.hypot(velocity_x, velocity_y)),
	}


def iter_frames(json_files):
	"""Yield (file_name, seconds, nanosec, objects) for every frame of every file.

	New files (one file = 5 minutes) contain several frames under 'frames'.
	Old files contain a single frame directly. Objects are converted to the old format.
	"""
	for json_file in json_files:
		json_data = get_json_file(json_file)
		frames = json_data.get('frames')
		if frames is None:
			frames = [json_data]
		for frame_data in frames:
			frame_time = frame_data.get('frame_timestamp')
			if frame_time is None:
				continue
			frame_objects = []
			for object_detect in frame_data.get('objects', []):
				object_detect = normalize_object_detect(object_detect)
				if object_detect.get('obj_id') is None or object_detect.get('obj_type') is None:
					continue
				frame_objects.append(object_detect)
			yield os.path.basename(json_file), frame_time['seconds'], frame_time.get('nanosec', 0), frame_objects


try:
	for source_file, time_stamp, nanosecond, frame_objects in iter_frames(loop):
		count += 1

		current_time_sec = float(time_stamp) + float(nanosecond) * 1e-9
		new_heading_history = {}
		frame_info = {}

		all_object_id = []
		for object_detect in frame_objects:
			object_position = get_possition(object_detect)
			if not inside_image(get_shifted_coordinate(object_position)):
				continue

			object_type = object_detect['obj_type']
			obj_id = object_detect['obj_id']
			all_object_id.append(obj_id)

			velocity = object_detect['velocity_magnitude']
			velocity_x = object_detect.get('velocity_x', 0.0)
			velocity_y = object_detect.get('velocity_y', 0.0)

			# heading from the velocity direction + angular velocity from the heading history
			heading, is_moving = motion_heading(object_detect['rotation_z'], velocity_x, velocity_y)
			previous_state = heading_history.get(str(obj_id))
			omega = compute_angular_velocity(heading, current_time_sec, is_moving, previous_state)
			history = list(previous_state) if previous_state else []
			history.append((current_time_sec, heading, is_moving))
			new_heading_history[str(obj_id)] = [s for s in history if current_time_sec - s[0] <= omega_window_sec + 0.5]

			frame_info[str(obj_id)] = {'vx': velocity_x, 'vy': velocity_y,
									'box_yaw': object_detect['rotation_z'], 'is_moving': int(is_moving)}

			# internal (prediction) frame: x <-> y swapped, yaw = -heading + pi/2
			yaw = -heading + np.pi/2
			velocity_ang = -omega
			width = object_detect['scale_y']
			height = object_detect['scale_x']
			# (time_stamp, nanosecond, x, y, v, yaw, velocity_ang, obj_type, width, height)
			objects.setdefault(str(obj_id), []).append((time_stamp, nanosecond, object_position[1],
														object_position[0], velocity,
														yaw, velocity_ang, object_type, width, height))

		# forget objects that are not in the current frame
		for obj_key in list(objects):
			if int(obj_key) not in all_object_id:
				del objects[obj_key]
		heading_history = new_heading_history

		p_t = predict_gaussians_for_objects(objects, delta_t=delta_t, class_params=class_motion)

		if count > 1:
			for i, zone in enumerate(zones):
				all_results = monte_carlo_all_pairs_collision_px_region(
					objects=objects,
					delta_t=delta_t,
					region_px=zone,
					n_samples=mc_samples,
					seed=42,
					preds=p_t)

				for pair, result in all_results.items():
					# skip pairs where both are pedestrians / cyclists
					if result['vehicle_1']['type'] in non_motorized and result['vehicle_2']['type'] in non_motorized:
						continue
					risk_level = classify_risk(result['collision_probability_given_region'])
					if risk_level == 'SAFE':
						continue

					# Save everything in the ORIGINAL sensor frame.
					row = {}
					for k, vehicle_key in ((1, 'vehicle_1'), (2, 'vehicle_2')):
						obj_key = result[vehicle_key]['id']
						pred = p_t[obj_key]
						veh = to_sensor_frame(result[vehicle_key], pred)
						for f, value in veh.items():
							row[f'object_{k}_{f}'] = value
						row.update({f'object_{k}_{f}': v for f, v in frame_info[obj_key].items()})
						row[f'object_{k}_omega'] = -objects[obj_key][-1][6]   # back to sensor frame
						row[f'object_{k}_sigma_a'] = pred['params']['sigma_a']
						row[f'object_{k}_sigma_alpha'] = pred['params']['sigma_alpha']
						row[f'object_{k}_track_frames'] = len(objects[obj_key])

					# ---------------- collision information ----------------
					row['collision_probability_global'] = result['collision_probability_global']
					row['collision_probability_given_region'] = result['collision_probability_given_region']
					row['time_stamp'] = time_stamp
					row['area_number'] = i + 1
					row['risk_level'] = risk_level
					row['distance'] = float(np.hypot(row['object_1_x'] - row['object_2_x'],
													row['object_1_y'] - row['object_2_y']))
					row['mean_continuous_risk'] = result['mean_continuous_risk']
					row['max_continuous_risk'] = result['max_continuous_risk']
					row['num_samples'] = result['num_total_samples']
					row['num_region_samples'] = result['num_region_samples']
					row['num_collision_samples'] = result['num_collision_samples']
					row['nanosec'] = nanosecond
					row['time_sec'] = f'{current_time_sec:.3f}'
					row['delta_t'] = delta_t
					row['source_file'] = source_file
					write_csv_row(row)
except KeyboardInterrupt:
	print('\nStopped by user (Ctrl+C).')
finally:
	csv_file.close()
	print(f'{csv_row_index} rows saved to {csv_path}')


print('------------------Done--------------------')
