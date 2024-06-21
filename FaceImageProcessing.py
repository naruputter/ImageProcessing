import numpy as np
import cv2 
import dlib
import os
import math
import glob
import json

IMAGE_FORMAT = ['jpg', 'jpeg', 'JPG', 'png', 'PNG']

DETECT_FACE_LANMARK_FILE_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_FILE_PATH    = "dlib_face_recognition_resnet_model_v1.dat"

detect_person_face   = dlib.get_frontal_face_detector()
detect_face_landmark = dlib.shape_predictor(DETECT_FACE_LANMARK_FILE_PATH)
face_recognition     = dlib.face_recognition_model_v1(FACE_RECOGNITION_FILE_PATH)

class FaceRecognition :

	def __init__(self, imagePath=None, faceDataList=None, faceJsonFilePath=None):

		if imagePath :
			self.set_image_path(imagePath)

		if faceDataList or faceJsonFilePath :
			self.set_face_database(faceDataList, faceJsonFilePath)

	def set_image_path( self, imagePath ):

		self.imagePath     = imagePath
		self.imageObject   = cv2.imread(imagePath)

	def set_face_database( self, jsonDataList=None, jsonFilePath=None ):

		return_data = { "code" : None, "data" : None, "desc" : None }

		if jsonDataList and jsonFilePath :

			raise Exception("set_face_database() can input only 1 parameter in [ jsonDataList, jsonFilePath ]")

		elif ( not jsonDataList ) and ( not jsonFilePath ) :

			raise Exception("set_face_database() must input 1 parameter in [ jsonDataList, jsonFilePath ]")

		else :

			if jsonFilePath :

				with open(jsonFilePath, "r") as json_file:
					self.FaceDatabase = json.load(json_file)

			elif jsonDataList :

				if type(jsonDataList) == dict :

					self.FaceDatabase = jsonDataList

				else :

					raise Exception("set_face_database() jsonDataList must only json")

	def reduce_image( self, targetSize=500000, minDimension=600 ):

		height, width = self.imageObject.shape[:2]

		if width < height:
		    scale_factor = minDimension / width
		else:
			scale_factor = minDimension / height

		new_width = int(width * scale_factor)
		new_height = int(height * scale_factor)

		resized_image = cv2.resize(self.imageObject, (new_width, new_height), interpolation=cv2.INTER_AREA)

		cv2.imwrite(self.imagePath, resized_image)


		image_size = os.path.getsize(self.imagePath)

		quality = 100

		while (targetSize <= image_size):
			
			quality = quality - 1
			cv2.imwrite(self.imagePath, self.imageObject, [cv2.IMWRITE_JPEG_QUALITY, quality])
			image_size = os.path.getsize(self.imagePath)

			if quality == 0 :

				break;

		self.imageObject = cv2.imread(self.imagePath)

	def extract_face( self, extendFrame=25, returnFaceImageArray=False ):

	    return_data = { "code" : None, "data" : None, "desc" : None }
	    detect_person_face = dlib.get_frontal_face_detector()

	    try :

	        image_height, image_width, channels = self.imageObject.shape

	        image_diagonal = math.sqrt( ( image_width**2 ) + ( image_height**2))

	        image_size_bytes = os.path.getsize(self.imagePath)

	        ### Find Brightness Value ###################################
	        gray = cv2.cvtColor(self.imageObject, cv2.COLOR_BGR2GRAY)
	        gray_image = cv2.cvtColor(self.imageObject, cv2.COLOR_BGR2GRAY)
	        image_brightness = np.mean(gray)/256

	        ### Find Center Point #######################################
	        image_center_point = [(image_width/2), (image_height/2)]

	        image_info = { "width" : image_width, "height" : image_height, "brightness":image_brightness, "fileSize": image_size_bytes }

	        face_frame_list = detect_person_face(self.imageObject, 1)

	        if len(face_frame_list) > 0:

	            face_feature_list = []

	            for face_frame in face_frame_list: 

	                face_feature = { "width" : None, "height" : None }

	                x_size = face_frame.right() - face_frame.left()
	                y_size = face_frame.top() - face_frame.bottom()

	                x_size_add = (x_size*extendFrame)/100
	                y_size_add = (y_size*extendFrame)/100

	                ### Find Center Point #######################################
	                face_center_point = [((face_frame.right() - face_frame.left())/2)+face_frame.left(), ((face_frame.bottom() - face_frame.top())/2)+face_frame.top()]
	                face_center_error_postition = distance_two_point(image_center_point, face_center_point)

	                face_frame_left   = int( face_frame.left()   - x_size_add )
	                face_frame_right  = int( face_frame.right()  + x_size_add )
	                face_frame_top    = int( face_frame.top()    + y_size_add*2 )
	                face_frame_bottom = int( face_frame.bottom() - y_size_add/2 )

	                if face_frame_left   < 0 : face_frame_left   = 0
	                if face_frame_right  < 0 : face_frame_right  = 0
	                if face_frame_top    < 0 : face_frame_top    = 0
	                if face_frame_bottom < 0 : face_frame_bottom = 0

	                xy = face_frame_left, face_frame_top
	                wh = face_frame_right, face_frame_bottom

	                crop_face_image = self.imageObject[face_frame_top:face_frame_bottom, face_frame_left:face_frame_right]

	                ### Find Brightness Value ###################################
	                gray_face = cv2.cvtColor(crop_face_image, cv2.COLOR_BGR2GRAY)
	                face_brightness = np.mean(gray_face) / 256

	                face_feature['brightness'] = face_brightness
	                face_feature['width']  = face_frame_right - face_frame_left
	                face_feature['height'] = face_frame_bottom - face_frame_top
	                face_feature['positionError'] = (face_center_error_postition/image_diagonal)*2

	                if returnFaceImageArray :
	                    face_feature['cropFaceImage'] = crop_face_image

	                face_feature_list.append(face_feature)

	            return_data['code'] = 200
	            return_data['data'] = {}
	            return_data['data']['imageInfo'] = image_info
	            return_data['data']['faceList'] = face_feature_list

	            return return_data

	        else :

	            return_data['code'] = 404
	            return_data['data'] = {}
	            return_data['data']['imageInfo'] = image_info
	            return_data['desc'] = "Not found face image"

	            return return_data


	    except Exception as e :

	        return_data['code'] = 400
	        return_data['desc'] = str(e)

	        return return_data

	def validate_face( self, smallestSize=10000, largestSize=10000000, lowestBright=0.4, highestErrorPostion=0.2, minFaceRatio=50 ) :

	    return_data = { "code" : None, "data" : None, "desc" : None }

	    extract_image_resp = self.extract_face( extendFrame=25, returnFaceImageArray=False )

	    if extract_image_resp['code'] == 200 :

	        image_info        = extract_image_resp['data']['imageInfo']
	        face_feature_list = extract_image_resp['data']['faceList']

	        if image_info['fileSize'] > largestSize :

	            return_data['code'] = -1
	            return_data['desc'] = "Image size is too large"

	        elif image_info['fileSize'] < smallestSize :

	            return_data['code'] = -1
	            return_data['desc'] = "Image size is too small"

	        elif len(face_feature_list) > 1 :

	            return_data['code'] = -1
	            return_data['desc'] = "Have many face in image"    

	        elif len(face_feature_list) == 1 :

	            face_feature = face_feature_list[0]

	            face_area = face_feature['width'] + face_feature['height']
	            image_area = image_info['width'] + image_info['height']

	            face_ratio = ( face_area / image_area ) * 100

	            if face_feature['brightness'] < lowestBright :

	                return_data['code'] = -1
	                return_data['desc'] = "Face brightness is to low"  

	            elif face_feature['positionError'] > highestErrorPostion :

	                return_data['code'] = -1
	                return_data['desc'] = "Face position not be center"    

	            elif face_ratio < minFaceRatio :

	                return_data['code'] = -1
	                return_data['desc'] = "Face in image is too small"    

	            else :

	                return_data['code'] = 200
	                return_data['desc'] = "success"

	        else :

	            return_data['code'] = 500
	            return_data['desc'] = "Unknow Case"   

	    else :

	        return_data['code'] = extract_image_resp['code']
	        return_data['desc'] = extract_image_resp['desc']

	    return return_data

	def detect_person( self, similarThreshold=0.60 ):

	    '''
	    jsonDataBase Example

	    {
	        "personName1" : ['0.1', 0.2, ... '0.1'], ### Get list from encode_face_from_image()
	        "personName2" : ['0.1', 0.2, ... '0.1'], ### list len 128
	    } 

	    '''

	    return_data = { "code" : None, "data" : None, "desc" : None }

	    img_rgb = cv2.cvtColor(self.imageObject, cv2.COLOR_BGR2RGB)
	    faces   = detect_person_face(img_rgb, 1)

	    if len(faces) <= 0:

	        return_data['code'] = 404
	        return_data['desc'] = "Face not found"

	    else:

	        person_list = []

	        for face in faces :

	            shape = detect_face_landmark(img_rgb, face)
	            face_descriptor = face_recognition.compute_face_descriptor(img_rgb, shape)

	            distances = [np.linalg.norm(face_descriptor - np.array(self.FaceDatabase[name])) for name in self.FaceDatabase]
	            min_index = np.argmin(distances)
	            min_distance = distances[min_index] 

	            similar_value = 1 - min_distance

	            if similar_value > similarThreshold:

	                person_data = {}
	                person_data['name'] = list(self.FaceDatabase.keys())[min_index]
	                person_data['similar'] = similar_value*100

	                person_list.append(person_data)

	        return_data['code'] = 200
	        return_data['data'] = person_list
	        return_data['desc'] = "Detect success"

	    return return_data


def read_image_in_folder_to_jsonDataBase(folderPath, savePath=None) :

    return_data = { "code" : None, "data" : None, "desc" : None}

    json_face_value_database = {}
    encode_result_list = []

    glob_path = f"{folderPath}/*"

    for file_path in glob.glob(glob_path):

        encode_result = { 'filename' : None, 'success':None }

        filename = file_path.replace((folderPath+'/'), "").split('.')[0]

        if file_path.split(".")[-1] in IMAGE_FORMAT :

            encode_face_resp = encode_face_from_image(file_path)

            if encode_face_resp['code'] == 200 :

                file_encode_list = encode_face_resp['data']

                if len(file_encode_list) > 1 :

                    encode_result['success']  = False
                    encode_result['filename'] = filename
                    encode_result['reason']   = "Have more 1 face in image"

                    encode_result_list.append(encode_result)


                else : 
                    
                    encode_result['success']  = True
                    encode_result['filename'] = filename

                    encode_result_list.append(encode_result)

                    ## Success ###
                    json_face_value_database[filename] = file_encode_list[0] 

            else :

                encode_result['success']  = False
                encode_result['filename'] = filename
                encode_result['reason']   = encode_face_resp['desc']

                encode_result_list.append(encode_result)

        else :

            encode_result['success']  = False
            encode_result['filename'] = filename
            encode_result['reason']   = "Not image file"

            encode_result_list.append(encode_result)


    if savePath and type(savePath) == str :

        with open(savePath, "w") as json_file:
            json.dump(json_face_value_database, json_file)

        return_data['code']   = 200
        return_data['data']   = encode_result_list
        return_data['desc']   = "Analyte image success"     

    else :

        return_data['code']   = 200
        return_data['data']   = json_face_value_database
        return_data['desc']   = "Analyte image success"

    return return_data

def encode_face_from_image(imagePath):

    return_data = return_data = { "code" : None, "data" : None, "desc" : None }

    try :

        imageObject = cv2.imread(imagePath)
        img_rgb = cv2.cvtColor(imageObject, cv2.COLOR_BGR2RGB)
        faces   = detect_person_face(img_rgb, 1)  # Upsample 1 time for better accuracy

        if len(faces) <= 0:

            return_data['code'] = 404
            return_data['desc'] = "Face not found"

        else:

            face_encode_list = []

            for face in faces :

                shape = detect_face_landmark(img_rgb, face)
                face_descriptor = face_recognition.compute_face_descriptor(img_rgb, shape)
                face_encode_list.append(list(face_descriptor))

            return_data['code'] = 200
            return_data['data'] = face_encode_list
            return_data['desc'] = "Return list of face array"

    except Exception as e:

        return_data['code'] = 500
        return_data['desc'] = str(e)

    return return_data


def distance_two_point(point_list_1, point_list_2):

    x1 = point_list_1[0]
    y1 = point_list_1[1]

    x2 = point_list_2[0]
    y2 = point_list_2[1]

    x_range_square = ( x2 - x1 ) ** 2
    y_range_square = ( y2 - y1 ) ** 2

    distance = math.sqrt(x_range_square + y_range_square)

    return distance


if __name__ == '__main__':
	
	# face_rec = FaceRecognition(imagePath="/Users/putter/Desktop/Image.jpeg", faceJsonFilePath="/Users/putter/Desktop/nvkDataset.json")

	# face_rec.set_image_path(imagePath="/Users/putter/Desktop/putter3.png")

	# print(read_image_in_folder_to_jsonDataBase("/Users/putter/Desktop/nvk"))
	face_rec = FaceRecognition(imagePath='/Users/putter/Desktop/sea.png')
	face_rec.reduce_image(500000)

	print(face_rec.validate_face())





