import cv2

def detect_face_from_image(imagePath, saveImagePath=None):

    '''
    1. Straight Face Image
    2. Face On Middle Of Picture
    3. Face Size More Than 50% Of Picture

    '''

    return_data = { "code" : None, "desc" : None, "count" : None }

    try:

        image = cv2.imread(imagePath)

        if image is not None:

            image = cv2.resize(image, (700, int((700*image.shape[0])/image.shape[1])))

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_image, 1.5, 4)

            if len(faces) <= 0 :

                return_data['code'] = -1
                return_data['desc'] = "Cannot Detect Face Image From Picture"
                return_data['count'] = len(faces)

                return return_data

            for (x, y, w, h) in faces:

                w_ad = int(w/2)
                h_ad = int(h/2)

                w = w + w_ad
                h = h + w_ad

                x = x - int(w_ad/2)
                y = y - int(h_ad/2)


                face_image = image[y:y+h, x:x+w]

                if saveImagePath :

                    resize_face_image = cv2.resize(face_image, (400, 400))
                    cv2.imwrite(saveImagePath , resize_face_image)

                return_data['code'] = 0
                return_data['desc'] = "Face Detected !"
                return_data['count'] = len(faces)

                return return_data

        elif not image :

            return_data['code'] = -1
            return_data['desc'] = "Image Not Found"

            return return_data

    except Exception as e :

        return_data['code'] = -1
        return_data['desc'] = str(e)

        return return_data

if __name__ == '__main__':
    
    print(detect_face_from_image("motorshow.jpg"))


