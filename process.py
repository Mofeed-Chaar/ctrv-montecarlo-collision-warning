import numpy as np
from config import parameters
import cv2

param = parameters.ModelParameters()
data = parameters.DataConfig()
colors = param.colors
img_size = param.img_size
colors_allarm = param.risk
def get_white_img():
	height,width = param.img_size

	white_image = np.ones((height, width, 3), dtype=np.uint8) * 255
	return white_image


def add_point(img,object_position,object_type,speed=None):
	img_size=img.shape
	point_x = img_size[0]/2 + object_position[0]*param.pixels_per_meter
	point_y = img_size[1]/2 + object_position[1]*param.pixels_per_meter
	img[int(point_x),int(point_y),:]=colors[object_type]

	return img


def get_possition(object_detected):
	position_x = object_detected['position_x']
	position_y = object_detected['position_y']
	return (position_x,position_y)



def drow_trajectory(start_point,end_point,img,object_type):
	return cv2.line(img, start_point, end_point, colors[object_type], 3)

def get_shifted_coordinate(point):

	x = point[0]
	y = point[1]
	point_x = img_size[0]/2 + x*param.pixels_per_meter
	point_y = img_size[1]/2 + y*param.pixels_per_meter
	return (int(point_x),int(point_y))



def put_titles_images(img):
	for i in  range(len(colors)):
		cv2.line(img, (img_size[0]-200,(i+1)*20), (img_size[0]-150,(i+1)*20), colors[list(colors.keys())[i]], 3)
		cv2.putText(img, list(colors.keys())[i], (img_size[0]-130,(i+1)*20+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
	return img

def inside_image(point):
	x = point[0]
	y = point[1]
	Check = True
	if x <0 or x>img_size[0] or y<0 or y>img_size[1]:
		Check = False

	return Check

def set_circle(img, center,radius=3,color=(0,0,0),thickness=-1,object_type=None):
	color=colors[object_type]
	cv2.circle(img, center, radius=radius, color=color, thickness=thickness)

def set_arrow(img, start_point, thickness=2,object_type=None,velocity=0,yaw=0):
	if velocity==0:
		return 0
	velocity=5*velocity
	#yaw = -yaw-np.pi/2#(yaw - np.pi) % (2*np.pi) + np.pi

	color=colors[object_type]
	end_point_x = int(velocity*np.cos(yaw))+start_point[0]
	end_point_y = int(velocity*np.sin(yaw))+start_point[1]
	end_point = (end_point_x,end_point_y)
	cv2.arrowedLine(img, start_point, end_point, color, thickness)

def drow_zone(img=None,zone=None):
	cv2.rectangle(img,(zone[0],zone[1]),(zone[2],zone[3]),(0,0,0),1)



def set_circle_allarm(img=None,allarm_type=None,obj1=None,obj2=None):
	radius = {'LOW':25,'MEDIUM':50,'HIGH':75,'CRITICAL':100}
	color = colors_allarm[allarm_type][::-1]



	position_obj1 = get_shifted_coordinate(obj1)
	position_obj2 = get_shifted_coordinate(obj2)

	cv2.circle(img,position_obj1, radius[allarm_type], color, 3) 
	cv2.circle(img,position_obj2, radius[allarm_type] ,color, 3)
	cv2.putText(img, allarm_type, (position_obj1[0]-30,position_obj1[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
	cv2.putText(img, allarm_type, (position_obj2[0]-30,position_obj2[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def put_allarm_text(img,list_allarm,shift=200):
	for i,text in enumerate(list_allarm):
		cv2.rectangle(img, (1080,i*20+shift-10), (1600,i*20+shift+10), (255, 255, 255), -1)
		cv2.putText(img, text , (1100,i*20+shift), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
