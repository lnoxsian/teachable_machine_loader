# this has the modules for importing and running a tflite model from teachable machine
# this code has been written by lnoxsian feel free to fork and do what you want 

from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np
import time

def disp_err_message(some_path):
    print(f"{some_path} is not found,check path name or just recheck it :)")

# this is the core func and classes

class lite_core:
    def do_transform_image(self, interpreter: Interpreter, image):
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def do_some_classification(self, interpreter: Interpreter, image, top_k=1):
        self.do_transform_image(interpreter, image)
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = np.squeeze(interpreter.get_tensor(output_details['index']))

        try:
            if output_details['dtype'] == np.uint8:
                scale, zero_point = output_details['quantization']
                output = scale * (output - zero_point)
        except:
            raise RuntimeError(
                """NotQuantizedModel: No match with uint8 data type.""")

        ordered = np.argpartition(-output, 1)
        return [(i, output[i]) for i in ordered[:top_k]][0]

# core callable class

class lite_modeler:
    def __init__(self, model_path="model.tflite", label_path="labels.txt"):
        self.model_path = model_path # the full model path
        self.label_path = label_path # the full label path

        try:
            self.lite_core = lite_core()
            self.interpreter = Interpreter(self.model_path)
            self.interpreter.allocate_tensors()
            print(f"{model_path} has been loaded")
        except RuntimeError as rterr:
            disp_err_message(self.model_path)
            raise (rterr)

    def _get_model_dets_(self):
        _, self.model_height, self.model_width, _ = self.interpreter.get_input_details()[0]['shape']

    def _get_label_dets_(self):
        try:
            with open(self.label_path, 'r') as label_file_obj:
                return [line.strip() for i, line in enumerate(label_file_obj.readlines())]
        except FileNotFoundError as fnferr:
            disp_err_message(self.label_path)
            raise (fnferr)

    def just_show_image(self, image_path="sample.jpeg"):
        self._get_model_dets_()
        image_obj = self.just_open_image_rgb(image_path)
        image_obj.show(f"{image_path}")

    def just_open_image_rgb(self, image_path="sample.jpeg"):
        self._get_label_dets_()
        try:
            return Image.open(image_path).convert('RGB').resize((self.model_width,self.model_height))
        except FileNotFoundError as fnferr:
            disp_err_message(image_path)
            raise (fnferr)

    def just_classify_imageo(self, image_obj):
        self._get_model_dets_()
        image_obj = image_obj.convert('RGB').resize((self.model_width,self.model_height))
        raw_result = self.lite_core.do_some_classification(self.interpreter, image_obj)
        return raw_result

    def just_classify_imagef(self, image_path="sample.jpeg"):
        self._get_model_dets_()
        image_obj = Image.open(image_path).convert('RGB').resize((self.model_width,self.model_height))
        raw_result = self.lite_core.do_some_classification(self.interpreter, image_obj)
        return raw_result

    def classify_an_imageo(self, image_obj):
        self._get_model_dets_()
        image_obj = image_obj.convert('RGB').resize((self.model_width,self.model_height))

        time_1 = time.time()
        label_id, prob = self.lite_core.do_some_classification(self.interpreter, image_obj)
        time_2 = time.time()
        classification_time = np.round(time_2-time_1, 3)

        labels = self._get_label_dets_()

        classification_label = labels[label_id].split()[1]
        classification_confidence = np.round(prob*100, 2)

        return {
            "id": label_id,
            "label": classification_label,
            "time": classification_time,
            "confidence": classification_confidence,
            "highest_class_id": label_id,
            "highest_class_prob": classification_confidence
        }

    def classify_an_imagef(self, image_path="sample.jpeg"):
        self._get_model_dets_()
        try:
            image_obj = Image.open(image_path).convert('RGB').resize((self.model_width,self.model_height))
        except FileNotFoundError as fnferr:
            disp_err_message(image_path)
            raise (fnferr)

        time_1 = time.time()
        label_id, prob = self.lite_core.do_some_classification(self.interpreter, image_obj)
        time_2 = time.time()
        classification_time = np.round(time_2-time_1, 3)

        labels = self._get_label_dets_()

        classification_label = labels[label_id].split()[1]
        classification_confidence = np.round(prob*100, 2)

        return {
            "id": label_id,
            "label": classification_label,
            "time": classification_time,
            "confidence": classification_confidence,
            "highest_class_id": label_id,
            "highest_class_prob": classification_confidence
        }
