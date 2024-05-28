import cv2 
import dlib
import os

def extract_face_from_image_path(imagePath, extendFrame=25, returnFaceImageArray=False):

    return_data = { "code" : None, "data" : None, "desc" : None }

    detect_person_face = dlib.get_frontal_face_detector()

    try:

        image = cv2.imread(imagePath)
        image_height, image_width, channels = image.shape
        image_size_bytes = os.path.getsize(imagePath)

        image_info = { "width" : image_width, "height" : image_height, "fileSize": image_size_bytes }

        face_frame_list = detect_person_face(image, 1)

        if len(face_frame_list) > 0:

            face_feature_list = []

            for face_frame in face_frame_list: 

                face_feature = { "width" : None, "height" : None }

                x_size = face_frame.right() - face_frame.left()
                y_size = face_frame.top() - face_frame.bottom()

                x_size_add = (x_size*extendFrame)/100
                y_size_add = (y_size*extendFrame)/100

                face_frame_left   = int( face_frame.left()   - x_size_add )
                face_frame_right  = int( face_frame.right()  + x_size_add )
                face_frame_top    = int( face_frame.top()    + y_size_add*2 )
                face_frame_bottom = int( face_frame.bottom() - y_size_add/2 )

                xy = face_frame_left, face_frame_top
                wh = face_frame_right, face_frame_bottom

                crop_face_image = image[face_frame_top:face_frame_bottom, face_frame_left:face_frame_right]

                face_feature['width']  = face_frame_right - face_frame_left
                face_feature['height'] = face_frame_bottom - face_frame_top

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


if __name__ == '__main__':
    
    resp_data = extract_face_from_image_path("806A0549.JPG")
    print(resp_data)
