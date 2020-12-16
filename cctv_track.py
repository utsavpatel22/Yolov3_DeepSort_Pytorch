import argparse
from sys import platform
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from yolov3.models import *  # set ONNX_EXPORT in models.py
from yolov3.utils.datasets import *
from yolov3.utils.utils import *
from deep_sort import DeepSort
import pickle
import sys
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from SocialDistancing.msg import BoxLocation, FloatArray
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
from itertools import combinations
import math
from squaternion import Quaternion
import json

deepsort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7")
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(image_width, image_height, bbox_left, bbox_top, bbox_w, bbox_h):
    """" Calculates the relative bounding box from absolute pixel values. """
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img, (x1, y1),(x2,y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

class Turtlebotgoal():
	def __init__(self):
		self.pose_subscriber = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.robot_pose_cb)
		self.robot_x = 0
		self.robot_y = 0
		self.robot_th = 0

	def robot_pose_cb(self, data):
		self.robot_x = data.pose.pose.position.x
		self.robot_y = data.pose.pose.position.y
		q = Quaternion(data.pose.pose.orientation.w, data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z)
		self.robot_th = q.to_euler(degrees=False)[2]



class Homography():
	def __init__(self):
		self.locked_identity = -1
		self.reached = True
		self.fixed_point = np.array([37, 460])
		self.offset_angle = 1.414
		self.ground_origin = np.array([-4.2, 1.86])
		self.r_theta_pub = rospy.Publisher("/target/position",Twist, queue_size = 10)

	def undistort(self, img, cal_dir='cal_pickle.p'):
	    #cv2.imwrite('camera_cal/test_cal.jpg', dst)
	    with open(cal_dir, mode='rb') as f:
	        file = pickle.load(f)
	    mtx = file['mtx']
	    dist = file['dist']
	    dst = cv2.undistort(img, mtx, dist, None, mtx)

	    return dst


	def four_point_transform(self, image, ped_data, tb_pose_obj):
	    # Change it with reorder function finally
	    # rect = np.array([(256, 115), (529, 90), (612, 286), (190, 305)], dtype = "float32")
	    rect = np.array([(152, 85), (433, 85), (598, 378), (38, 418)], dtype = "float32")
	    # rect = order_points(pts)
	    (tl, tr, br, bl) = rect
	    # print(tl, tr, br, bl)
	    scale = 2
	    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	    maxWidth = scale * max(int(widthA), int(widthB))
	    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	    maxHeight = scale * max(int(heightA), int(heightB))
	    # dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")
	    dst = np.array([(37, 63), (337, 63), (337, 460), (37, 460)], dtype = "float32")
	    
	    # compute the perspective transform matrix and then apply it
	    M = cv2.getPerspectiveTransform(rect, dst)
	    warped = cv2.warpPerspective(image, M, (640, 480))

	    # if (len(ped_data.keys()) >= 2):
	    # 	ped_combinations = list(combinations(ped_data.keys(),2))
	    	# for k in range(len(ped_combinations)):
	    	# 	point1 = np.matmul(M, ped_data[ped_combinations[k][0]])
	    	# 	point2 = np.matmul(M, ped_data[ped_combinations[k][1]])
	    	# 	point1[0] = point1[0]/ point1[2]
	    	# 	point1[1] = point1[1]/ point1[2]

	    	# 	point2[0] = point2[0]/ point2[2]
	    	# 	point2[1] = point2[1]/ point2[2] 
	    	# 	cv2.circle(warped,(int(point1[0]), int(point1[1])), 5, (255,0,255), -1)
	    	# 	cv2.circle(warped,(int(point2[0]), int(point2[1])), 5, (255,0,255), -1)
	    	# 	print("The points are {} ".format(ped_combinations[k][0]))
	    	# 	print("The other point is {}".format(ped_combinations[k][1]))
	    	# 	ped_distance = self.find_distance(point1, point2)
	    	# 	print("The distance between ped-{} and ped-{} is {}".format(ped_combinations[k][0],ped_combinations[k][1], ped_distance))
	    # return the warped image

	    list_set = []
	    if (len(ped_data.keys()) > 1):
		    ped_combinations = list(combinations(ped_data.keys(),2))
		    non_compliant_ped = []
		    for k in range(len(ped_combinations)):
	    		point1 = np.matmul(M, ped_data[ped_combinations[k][0]])
	    		point2 = np.matmul(M, ped_data[ped_combinations[k][1]])
	    		point1[0] = point1[0]/ point1[2]
	    		point1[1] = point1[1]/ point1[2]

	    		point2[0] = point2[0]/ point2[2]
	    		point2[1] = point2[1]/ point2[2] 
	    		# cv2.circle(warped,(int(point1[0]), int(point1[1])), 5, (255,0,255), -1)
	    		# cv2.circle(warped,(int(point2[0]), int(point2[1])), 5, (255,0,255), -1)
	    		# print("The points are {} ".format(ped_combinations[k][0]))
	    		# print("The other point is {}".format(ped_combinations[k][1]))
	    		ped_distance = self.find_distance(point1, point2)
	    		# print("The distance between ped-{} and ped-{} is {}".format(ped_combinations[k][0],ped_combinations[k][1], ped_distance))

		    	if(ped_distance < 2):
			    	non_compliant_ped.append(ped_combinations[k])

		    if(len(non_compliant_ped) > 0):
		    	list_set.append(set(non_compliant_ped[0]))
		    	for i in range(1, len(non_compliant_ped)):
		    		count = 0
		    		for j in range(0, len(list_set)):
		    			int_len = len(list_set[j].intersection(non_compliant_ped[i]))
		    			if(int_len > 0):
		    				list_set[j] = list_set[j].union(set(non_compliant_ped[i]))
		    			else:
		    				count += 1
		    		if(count == len(list_set)):
		    			list_set.append(set(non_compliant_ped[i]))

		    # print("The groups {}".format(list_set))
		    if (len(list_set) > 0):
		    	largest_grp = max(list_set, key=len)
		    	# print("The largest group is {}".format(largest_grp))
		    	min_cent_dist = image.shape[0]
		    	min_dist_identity = -1
		    	for identity in largest_grp:
		    		tmp_cent_dist = abs((image.shape[1]/2) - ped_data[identity][0])
		    		# print("The image width {}".format(image.shape[1]))
		    		# print("The center point {} for identity {}".format(ped_data[identity][0], identity))
		    		if(tmp_cent_dist < min_cent_dist):
		    			min_cent_dist = tmp_cent_dist
		    			min_dist_identity = identity
		    			self.locked_identity = min_dist_identity
		    			self.reached = False

		    	point_ground_norm = np.matmul(M, ped_data[self.locked_identity])
		    	point_ground_norm[0] = point_ground_norm[0] / point_ground_norm[2]
		    	point_ground_norm[1] = point_ground_norm[1] / point_ground_norm[2]

		    	r_ground = self.find_distance(self.fixed_point, point_ground_norm) 
		    	cv2.circle(warped,(int(point_ground_norm[0]), int(point_ground_norm[1])), 5, (255,0,255), -1)
		    	theta_ground = -1 * np.arctan([(point_ground_norm[1]-self.fixed_point[1]) / (point_ground_norm[0]-self.fixed_point[0])])
		    	theta_map = theta_ground - self.offset_angle

		    	x_map = self.ground_origin[0] + r_ground * math.cos(theta_map)
		    	y_map = self.ground_origin[1] + r_ground * math.sin(theta_map)
		    	# print("The locked identity data {}".format(ped_data[self.locked_identity]))
		    	# print("The locked identity is {}".format(self.locked_identity))
		    	# print("The theta ground {}".format(theta_ground * (180/np.pi)))
		    	# print("The current turtlebot pose {} -- {} -- {}".format(tb_pose_obj.robot_x, tb_pose_obj.robot_y, tb_pose_obj.robot_th))
		    	# print("The x_map and y_map {} --- {}".format(x_map, y_map))

		    	r_robot = (math.sqrt((x_map - tb_pose_obj.robot_x)**2 + (y_map - tb_pose_obj.robot_y)**2))
		    	ang_uncorrected = np.arctan([(y_map - tb_pose_obj.robot_y) / (x_map - tb_pose_obj.robot_x)])
		    	# print("The robot relative angle is {}".format(robot_rel_ang * (180/np.pi)))

		    	# Translating the map axis to the robot
		    	x_shift_robot = x_map - tb_pose_obj.robot_x
		    	y_shift_robot = y_map - tb_pose_obj.robot_y

		    	if x_shift_robot > 0:
		    		ang_corrected = ang_uncorrected

		    	elif x_shift_robot < 0:
		    		if y_shift_robot > 0: # 2nd co
		    			ang_corrected = ang_uncorrected + np.pi
		    		elif y_shift_robot < 0: # 3rd co
		    			ang_corrected = ang_uncorrected - np.pi

		    	theta_rotate = ang_corrected - tb_pose_obj.robot_th
		    	print("theta rotate {}".format(theta_rotate * (180/np.pi)))

		    	# print("Theta_robot before if condition {}".format(theta_robot * (180/np.pi)))
		    	if (theta_rotate > np.pi):
		    		theta_rotate = theta_rotate - (2*np.pi) 
		    	elif (theta_rotate < -np.pi):
		    		theta_rotate = theta_rotate + (2*np.pi)
		    	print("theta rotate after pi correction {}".format(theta_rotate * (180/np.pi)))
		    	# print("The robot distance {} and angle {}".format(r_robot, theta_robot*(180/np.pi)))
		    	
		    	if (r_robot < 1):
		    		self.reached = True
		    		# print("The reached status {}".format(self.reached))

		    	if (self.reached == False):
		    		r_theta_obj = Twist()
		    		r_theta_obj.linear.x = r_robot
		    		r_theta_obj.linear.y = -theta_rotate
		    		self.r_theta_pub.publish(r_theta_obj)

		    	else:
		    		r_theta_obj = Twist()
		    		r_theta_obj.linear.x = 0
		    		r_theta_obj.linear.y = 0
		    		self.r_theta_pub.publish(r_theta_obj)

		    # elif (len(list_set) == 0) and not self.reached and (list(self.ped_data_dict.keys()).count(self.locked_identity) != 0):
			# 	r_theta = Twist()
			# 	r_theta.linear.x = self.ped_data_dict[self.locked_identity][2] / 1000
			# 	r_theta.linear.y = self.ped_data_dict[self.locked_identity][3] * (3.14/180)
			# 	self.r_theta_pub.publish(r_theta)

			# 	if (self.ped_data_dict[self.locked_identity][2] < 1000):
			# 		self.reached = True
			# else:
			# 	r_theta = Twist()
			# 	r_theta.linear.x = 0
			# 	r_theta.linear.y = 0
			# 	self.r_theta_pub.publish(r_theta)
			# 	self.locked_identity = -1
			# 	self.reached = False


		    else:
		    	r_theta = Twist()
		    	r_theta.linear.x = 0
		    	r_theta.linear.y = 0
		    	self.r_theta_pub.publish(r_theta)
		    	self.locked_identity = -1
	    print("Reached status {}".format(self.reached))
		# self.ped_data_dict = {}

	    return warped

	def find_distance(self, point1, point2):
		scale = 0.00963
		return (math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) * scale)




def detect(save_img=True):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or not (source == 'intelcam') or isinstance(source, int) or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    intelcam = source == 'intelcam'

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Defining publisher object
    ped_data_pub = rospy.Publisher("/pedestrian_data",BoxLocation, queue_size = 10)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_img = False
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)

    elif intelcam:
        save_img = False
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadIntelCam(img_size=img_size, half=half)

    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()

    # dictionary to load into json
    jsonDictionary = {}
    frame_id = 0  

    tb_pose_obj = Turtlebotgoal()
    for path, img, im0s, vid_cap in dataset:
        if img is None:
           break
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            elif intelcam:
                p, s, im0 = path[i], '%g: ' % i, im0s
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # print(det[:, :5])
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Write results
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape  # get image shape
                    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
                    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
                    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
                    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, bbox_left, bbox_top, bbox_w, bbox_h)
                    #print(x_c, y_c, bbox_w, bbox_h)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    label = '%s %.2f' % (names[int(cls)], conf)
                    #
                    #print('bboxes')
                    #print(torch.Tensor(bbox_xywh))
                    #print('confs')
                    #print(torch.Tensor(confs))
                    outputs = deepsort.update((torch.Tensor(bbox_xywh)), (torch.Tensor(confs)) , im0)
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        ped_data = {}
                        ped_data_tmp = {}
                        for l in range(bbox_xyxy.shape[0]):
                        	tmp_array = np.expand_dims(np.asarray([(bbox_xyxy[l][0] + bbox_xyxy[l][2]) / 2, bbox_xyxy[l][3], 1]), axis=1) # [x_center, max y, 1]
                        	tmp_array_list = tmp_array.tolist()
                        	# cv2.circle(im0,(int((bbox_xyxy[l][0] + bbox_xyxy[l][2]) / 2), int(bbox_xyxy[l][3])), 5, (0,0,255), -1)
                        	ped_data[int(identities[l])] =  tmp_array
                        	ped_data_tmp[int(identities[l])] = tmp_array_list
                        jsonDictionary[frame_id] = ped_data_tmp
                        # print("The ped dictionary {}".format(ped_data))
                        hg_obj = Homography()
                        undistort_img = hg_obj.undistort(im0)
                        warped_img = hg_obj.four_point_transform(undistort_img, ped_data, tb_pose_obj)
                        cv2.imshow('Undistort_img',undistort_img)
                        cv2.imshow('warped_img',warped_img)
                        
                        # print("The location of bounding boxes {}".format(bbox_xyxy))
                        # print("The location of bounding boxes {}".format(bbox_xyxy[0][1]))
                        # print("The id {}".format(identities))
                        draw_boxes(im0, bbox_xyxy, identities)
                    #print('\n\n\t\ttracked objects')
                    #print(outputs)


            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
        frame_id += 1
    textFile = open("json.txt","w+")
    jsonOutput=json.dumps(jsonDictionary, indent = 4)
    textFile.write(jsonOutput)
    textFile.close()

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov3/cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='yolov3/data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='yolov3/weights/yolov3-spp-ultralytics.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='0', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    rospy.init_node('ped_tracker', anonymous=True)

    with torch.no_grad():
        detect()
