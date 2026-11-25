import glob
import os
import json
import tqdm
import cv2
import copy
from config import parameters
from utils import (get_json_file
	)


from proccess import (get_white_img,
	add_point,
	get_possition,
	drow_trajectory,
	get_shifted_coordinate,
	put_titles_images,
	inside_image,
	set_circle,
	set_arrow
	)

from trajectory import (
	predict_gaussians_for_objects
	)

from drow_gaussian import (
	draw_gaussian_heatmap_world_on_image
	)


param = parameters.ModelParameters()
data = parameters.DataConfig()
#--------------------------------------------------------------for delete

ob_is_test = None
max_num_object = 150

num_test = 1

#------------------------------------------------------------


data_path = data.data_path
data_paths = glob.glob(os.path.join(data_path,'*.json'))
data_paths.sort()#(reverse=True)
print(f'Number of files{len(data_paths)}')

loop = tqdm.tqdm(data_paths)
white_img = get_white_img()
white_img = put_titles_images(white_img)
count=0
objects = {}


filter_object_list = []
for json_file in loop:
	count+=1


	
	json_data = get_json_file(json_file)

	#print(json_data)

	time = json_data['frame_timestamp']
	time_stamp = time['seconds']
	nanosecond = time['nanosec']
	num_object=0

	#print(time_stamp)
	for object_detect in json_data['objects']:
		#print(object_detect)
		
		num_object+=1
		object_position = get_possition(object_detect)
		if not inside_image(get_shifted_coordinate(object_position)):

			continue


		
		object_type = object_detect['obj_type']





		white_img = add_point(white_img,object_position,object_type)
		



cv2.imwrite("sample/white_image.png", white_img)
print('------------------Done--------------------')

