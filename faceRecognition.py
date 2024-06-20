import dlib
import cv2
import numpy as np
import glob, os
import json

IMAGE_FORMAT = ['jpg', 'jpeg', 'JPG', 'png', 'PNG']

DETECT_FACE_LANMARK_FILE_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_FILE_PATH    = "dlib_face_recognition_resnet_model_v1.dat"

detect_person_face   = dlib.get_frontal_face_detector()
detect_face_landmark = dlib.shape_predictor(DETECT_FACE_LANMARK_FILE_PATH)
face_recognition     = dlib.face_recognition_model_v1(FACE_RECOGNITION_FILE_PATH)

def encode_face_from_image(imagePath):

    return_data = return_data = { "code" : None, "data" : None, "desc" : None }

    try :

        img     = cv2.imread(imagePath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces   = detect_person_face(img_rgb, 1)  # Upsample 1 time for better accuracy

        if len(faces) <= 0:

            return_data['code'] = 404
            return_data['desc'] = "Face not found"

        else:

            face_encode_list = []

            for face in faces :

                shape = detect_face_landmark(img_rgb, face)
                print(shape)
                face_descriptor = face_recognition.compute_face_descriptor(img_rgb, shape)
                face_encode_list.append(list(face_descriptor))

            return_data['code'] = 200
            return_data['data'] = face_encode_list
            return_data['desc'] = "Return list of face array"

    except Exception as e:

        return_data['code'] = 500
        return_data['desc'] = str(e)

    return return_data


def detect_person_from_image(imagePath, jsonDataBase, similar_threshold=0.5):

    '''
    jsonDataBase Example

    {
        "personName1" : ['0.1', 0.2, ... '0.1'], ### Get list from encode_face_from_image()
        "personName2" : ['0.1', 0.2, ... '0.1'], ### list len 128
    } 

    '''

    return_data = { "code" : None, "data" : None, "desc" : None }

    img     = cv2.imread(imagePath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces   = detect_person_face(img_rgb, 1)

    if len(faces) <= 0:

        return_data['code'] = 404
        return_data['desc'] = "Face not found"

    else:

        person_list = []

        for face in faces :

            shape = detect_face_landmark(img_rgb, face)
            face_descriptor = face_recognition.compute_face_descriptor(img_rgb, shape)

            distances = [np.linalg.norm(face_descriptor - np.array(jsonDataBase[name])) for name in jsonDataBase]
            min_index = np.argmin(distances)
            min_distance = distances[min_index] 

            similar_value = 1 - min_distance

            if similar_value > similar_threshold:

                person_data = {}
                person_data['name'] = list(database.keys())[min_index]
                person_data['similar'] = similar_value*100

                person_list.append(person_data)

        return_data['code'] = 200
        return_data['data'] = person_list
        return_data['desc'] = "Face not found"

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
            

if __name__ == '__main__':

    # print(read_image_in_folder_to_jsonDataBase('nvk', savePath="test.json"))

    print(encode_face_from_image("/Users/putter/Desktop/putter.jpg"))

    # with open("test.json", "r") as json_file:
      # database = json.load(json_file)

    # print(database)

    # print("==========================================")

    # print(detect_person_from_image('putter.png', database))

 
