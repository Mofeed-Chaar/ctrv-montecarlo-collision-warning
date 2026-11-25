"""Live view: trajectory prediction and collision alarms on the intersection map.

Run:  python visualization.py        (press q to stop)
"""
import glob
import os
import copy
from datetime import datetime

import tqdm
import cv2
import numpy as np
from config import parameters
from utils import get_json_file


from proccess import (
	get_possition,
	get_shifted_coordinate,
	put_titles_images,
	inside_image,
	set_circle,
	set_arrow,
	drow_zone,
	set_circle_allarm,
	put_allarm_text
	)
from collision_predict import monte_carlo_all_pairs_collision_px_region
from trajectory import (
	predict_gaussians_for_objects
	)

from drow_gaussian import (
	draw_gaussian_heatmap_world_on_image
	)


param = parameters.ModelParameters()
data = parameters.DataConfig()

zones = param.zones
delta_t = param.delta_t


print(f'Number of zones is {len(zones)}')

non_motorized = ['Pedestrian','Cyclist']
max_frames = getattr(param, 'max_frames', None)
mc_samples = getattr(param, 'mc_samples_visualization', 50)

data_path = data.data_path
data_paths = glob.glob(os.path.join(data_path,'*.json'))
data_paths.sort()
print(f'Number of files {len(data_paths)}')

loop = tqdm.tqdm(data_paths)
count=0
objects = {}
background_path = data.background_image
if cv2.imread(background_path) is None:
	raise FileNotFoundError(f'Background image not found: {background_path}')

allarm_text=[]


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

	previous_state: list of (time_sec, heading, is_moving) of the earlier frames of
	this object (or None). Differencing two consecutive frames (0.1 s) is dominated
	by noise, so a window is used instead. Returns 0.0 when there is not enough data.
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

	slope = np.polyfit(t - t[0], h, 1)[0]
	return float(slope)


def normalize_object_detect(object_detect, previous_state=None, current_time_sec=None):
	"""Convert the new aggregated-frame object format to the old format used by the rest of this script.

	Old format example:
		obj_id, obj_type, position_x, position_y, scale_x, rotation_z, velocity_magnitude, ...

	New format example inside each frame:
		track_id, class_name, bounding_box.position, bounding_box.scale, bounding_box.yaw, velocity.x/y
	"""
	if 'obj_id' in object_detect and 'obj_type' in object_detect:
		return object_detect

	bounding_box = object_detect.get('bounding_box', {})
	position = bounding_box.get('position', {})
	scale = bounding_box.get('scale', {})
	velocity = object_detect.get('velocity', {})

	velocity_x = float(velocity.get('x', object_detect.get('velocity_x', 0.0)))
	velocity_y = float(velocity.get('y', object_detect.get('velocity_y', 0.0)))
	box_yaw = float(bounding_box.get('yaw', object_detect.get('rotation_z', 0.0)))

	# rotation_z is the heading used by the motion model (velocity direction when moving).
	rotation_z, is_moving = motion_heading(box_yaw, velocity_x, velocity_y)

	if 'velocity_angular' in object_detect:
		velocity_angular = float(object_detect['velocity_angular'])
	else:
		velocity_angular = compute_angular_velocity(
			current_heading=rotation_z,
			current_time_sec=current_time_sec,
			is_moving=is_moving,
			previous_state=previous_state
		)

	normalized_object = dict(object_detect)
	normalized_object.update({
		'obj_id': object_detect.get('obj_id', object_detect.get('track_id')),
		'obj_type': object_detect.get('obj_type', object_detect.get('class_name')),
		'position_x': position.get('x', object_detect.get('position_x', 0.0)),
		'position_y': position.get('y', object_detect.get('position_y', 0.0)),
		'position_z': position.get('z', object_detect.get('position_z', 0.0)),
		'scale_x': scale.get('x', object_detect.get('scale_x', 0.0)),
		'scale_y': scale.get('y', object_detect.get('scale_y', 0.0)),
		'scale_z': scale.get('z', object_detect.get('scale_z', 0.0)),
		'rotation_x': object_detect.get('rotation_x', 0.0),
		'rotation_y': object_detect.get('rotation_y', 0.0),
		'rotation_z': rotation_z,
		'box_yaw': box_yaw,
		'is_moving': is_moving,
		'velocity_x': velocity_x,
		'velocity_y': velocity_y,
		'velocity_angular': velocity_angular,
		'velocity_magnitude': float(np.hypot(velocity_x, velocity_y)),
	})
	return normalized_object


previous_frame_object_state = {}
stop_requested = False
for json_file in loop:
	json_data = get_json_file(json_file)

	# Old files contain one frame directly.
	# New files contain several frames under json_data['frames'].
	frames = json_data.get('frames')
	if frames is None:
		frames = [json_data]

	for frame_data in frames:
		count += 1

		trajectory_img = cv2.imread(background_path)
		trajectory_img = put_titles_images(trajectory_img)

		for zone in zones:
			drow_zone(trajectory_img,zone)


		time = frame_data.get('frame_timestamp')
		if time is None:
			continue

		time_stamp = time['seconds']
		nanosecond = time.get('nanosec', 0)
		current_time_sec = float(time_stamp) + float(nanosecond) * 1e-9
		current_frame_object_state = {}

		dt = datetime.fromtimestamp(time_stamp)
		day_time = dt.strftime('%H:%M:%S')

		cv2.putText(trajectory_img, 'Date:'+str(dt), (900,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

		num_object=0

		all_object_id = []
		for object_detect in frame_data.get('objects', []):
			raw_obj_id = object_detect.get('obj_id', object_detect.get('track_id'))
			previous_state = previous_frame_object_state.get(str(raw_obj_id))
			object_detect = normalize_object_detect(
				object_detect,
				previous_state=previous_state,
				current_time_sec=current_time_sec
			)

			if object_detect.get('obj_id') is None or object_detect.get('obj_type') is None:
				continue

			num_object+=1
			object_position = get_possition(object_detect)
			if not inside_image(get_shifted_coordinate(object_position)):
				continue

			object_type = object_detect['obj_type']
			obj_id = object_detect['obj_id']
			all_object_id.append(obj_id)

			velocity= object_detect['velocity_magnitude']
			yaw = -object_detect['rotation_z']+np.pi/2  # internal frame
			velocity_ang = -object_detect['velocity_angular']
			width = object_detect['scale_y']
			height = object_detect['scale_x']
			objects.setdefault(str(obj_id),[]).append((time_stamp,nanosecond,object_position[1], 
													object_position[0],velocity,
													yaw,velocity_ang,object_type,width,height))#(time_stamp,nanosecond, x, y, v, yaw, velocity_ang, obj_type)

			# keep a short heading history per object for the angular-velocity fit
			history = list(previous_state) if previous_state else []
			history.append((current_time_sec, object_detect['rotation_z'], object_detect.get('is_moving', True)))
			history = [s for s in history if current_time_sec - s[0] <= omega_window_sec + 0.5]
			current_frame_object_state[str(obj_id)] = history

		previous_frame_object_state = current_frame_object_state
		objects_itt = copy.deepcopy(objects)
		for object_in_last_scine in objects_itt:
			if int(object_in_last_scine) in all_object_id:
				continue
			del objects[object_in_last_scine]
		del objects_itt

		p_t=predict_gaussians_for_objects(objects, delta_t=delta_t, class_params=class_motion)

		if count>1:
			for i,zone in enumerate(zones):

				all_results = None
				all_results = monte_carlo_all_pairs_collision_px_region(
				objects=objects,
				delta_t=delta_t,
				region_px=zone,
				n_samples=mc_samples,
				seed=42,
				preds = p_t)


				for pair, result in all_results.items():

					# skip pairs where both are pedestrians / cyclists
					if result['vehicle_1']['type'] in non_motorized and result['vehicle_2']['type'] in non_motorized:
						continue
					if result['risk_level'] != 'SAFE':
						print(f"[{day_time}] zone {i+1}: {result['risk_level']} "
							f"{result['vehicle_1']['type']} {result['vehicle_1']['id']} - "
							f"{result['vehicle_2']['type']} {result['vehicle_2']['id']}, "
							f"p = {result['collision_probability_global']:.3f}")

						set_circle_allarm(img=trajectory_img,allarm_type=result['risk_level'],obj1=result['vehicle_1']['center_current'],obj2=result['vehicle_2']['center_current'])
						
						allarm_text.append(result['vehicle_1']['type']+' and '+result['vehicle_2']['type']+' Zone Number '+str(i+1) +' Probability: '+str(result['collision_probability_global'])+ ', Time:'+day_time)
						if len(allarm_text)>5:
							allarm_text.pop(0)


			for item in objects: # item is the obj_id
				one_object = objects[item]
				if len(one_object)<2:
					continue

				end_point_x = one_object[-1][2]
				end_point_y = one_object[-1][3]
				end_point = (end_point_x,end_point_y)

				# mean of the last (up to) 3 frames; yaw averaged on the circle
				# (a plain mean of e.g. +3.1 and -3.1 rad would give 0)
				last = one_object[-3:]
				velocity = float(np.mean([r[4] for r in last]))
				yaw = float(np.arctan2(np.mean([np.sin(r[5]) for r in last]),
									np.mean([np.cos(r[5]) for r in last])))

				object_type = one_object[0][7]

				end_point = get_shifted_coordinate(end_point)
				set_circle(img=trajectory_img, center=end_point,object_type=object_type)
				set_arrow(img=trajectory_img, start_point=end_point, thickness=2,object_type=object_type,velocity=velocity,yaw=yaw)

				# ---------------- trajectory prediction ----------------

				predict_object= p_t[item] 
				mu=predict_object["mu"]
				sigma = predict_object["Sigma"]
				trajectory_img = draw_gaussian_heatmap_world_on_image(
					trajectory_img,
					mu_world=mu,
					Sigma_world=sigma,
					s=16.0,
					offset_px=(800, 800),
					flip_y=False,
					confidence=0.99,
					alpha=0.60,
					draw_ellipse=False,
					add_colorbar=True
					)
			put_allarm_text(trajectory_img,allarm_text)


			cv2.imshow("Image", trajectory_img)
			key = cv2.waitKey(500)	
			if key == ord('q'):
				stop_requested = True
				break

			if max_frames is not None and count >= max_frames:
				stop_requested = True
				break

		if stop_requested:
			break

	if stop_requested:
		break


print('------------------Done--------------------')
