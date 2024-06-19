import cv2 
import dlib
import os
import math
import numpy as np

def distance_two_point(point_list_1, point_list_2):

    x1 = point_list_1[0]
    y1 = point_list_1[1]

    x2 = point_list_2[0]
    y2 = point_list_2[1]

    x_range_square = ( x2 - x1 ) ** 2
    y_range_square = ( y2 - y1 ) ** 2

    distance = math.sqrt(x_range_square + y_range_square)

    return distance

def extract_face_from_image_path(imagePath, extendFrame=25, returnFaceImageArray=False):

    return_data = { "code" : None, "data" : None, "desc" : None }

    detect_person_face = dlib.get_frontal_face_detector()

    try :

        image = cv2.imread(imagePath)
        image_height, image_width, channels = image.shape

        image_diagonal = math.sqrt( ( image_width**2 ) + ( image_height**2))

        image_size_bytes = os.path.getsize(imagePath)

        ### Find Brightness Value ###################################
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_brightness = np.mean(gray)/256

        ### Find Center Point #######################################
        image_center_point = [(image_width/2), (image_height/2)]

        image_info = { "width" : image_width, "height" : image_height, "brightness":image_brightness, "fileSize": image_size_bytes }

        face_frame_list = detect_person_face(image, 1)

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

                crop_face_image = image[face_frame_top:face_frame_bottom, face_frame_left:face_frame_right]

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

def vaildate_face_from_image_path(imagePath, smallestSize=10000, largestSize=1000000, lowestBright=0.4, highestErrorPostion=0.2, minFaceRatio=50 ) :

    return_data = { "code" : None, "data" : None, "desc" : None }

    extract_image_resp = extract_face_from_image_path(imagePath, extendFrame=25, returnFaceImageArray=False)

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


if __name__ == '__main__':
    
    extract_resp = extract_face_from_image_path("moo.jpg")

    print(extract_resp)

    validate_resp = vaildate_face_from_image_path("moo.jpg") 

    print(validate_resp)
