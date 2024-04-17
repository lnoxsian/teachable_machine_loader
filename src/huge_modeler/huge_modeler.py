# this is the python module for importing teachable machine tensorflow models
from tensorflow.keras.models import load_model
from PIl import Image,ImageOps
import numpy as np
import os

def disp_err(some_err):
    print(f"{some_err} is not found please do check for path or name")

class huge_modeler:
    def __init__(self, model_path="keras_model.h5", label_path="label_path"):
        self.model_path = model_path
        self.label_path = label_path
        self.model_type = os.path.splitext(model_path)[1].lower()

        np.set_printoptions(suppress=True)
        try:
            self.model = load_model(model_path,compile=False)
        except IOError as e:
            disp_err(self.model_path)
            raise IOError from e
        except:
            disp_err(model_path)
        try:
            self.label_data = open(self.label_path,'r').readlines()
        except IOError as e:
            disp_err(label_path)
            raise IOError from e
        except:
            disp_err(label_path)
            raise FileNotFoundError

        self.is_model_created= self.model_type in ['keras', 'Keras','h5','h5py']
        if self.is_model_created:
            print(f"{model_path} object creation success")
        else:
            print(f"{self.model_type} is not supported")

    def proc_image(self, image):

        image_data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
        crop_size = (224,224)
        image_obj = ImageOps.fit(image,crop_size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image_obj)

        image_array_normalized = (image_array.astype(np.float32) / 127.0) -1

        image_data[0] = image_array_normalized
        return image_data

    def get_image_classification(self, image):

        img_data = self.proc_image(image)
        prediction = self.model.predict(img_data)
        class_index = np.argmax(prediction)
        class_name = self.label_data[class_index]
        class_confidence = prediction[0][class_index]

        return {
                 "class_name": class_name[2:],
                 "highest_class_name": class_name[2:],
                 "highest_class_id": class_index,
                 "class_index": class_index,
                 "class_id": class_index,
                 "predictions": prediction,
                 "all_predictions": prediction,
                 "class_confidence": class_confidence,
                 "highest_class_confidence": class_confidence,
        }

    def just_classify_imagef(self, image_path="sample.jpeg"):
        try:
            image_obj = Image.open(image_path)
            if frame.mode != RGB:
                image_obj = image_obj.convert("RGB")
        except FileNotFoundError as fnferr:
             disp_err(image_path)
             raise fnferr
        except TypeError as terr:
             print("Cannot convert to RGB")
             raise terr
        try:
            if self.is_model_created:
                return self.get_image_classification(image_obj)
        except BaseException as bexp:
            print("some error in classification")
            raise bexp
