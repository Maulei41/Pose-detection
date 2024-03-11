import tensorflow as tf
import numpy as np

class Classifier(object):
    def __init__(self, model_name : str, label_file_name : str):
        interperter = tf.lite.Interpreter(model_path = model_name)
        interperter.allocate_tensors()
        self.input_detail = interperter.get_input_details()[0]["index"]
        self.output_detail = interperter.get_output_details()[0]["index"]
        self.interperter = interperter
        self.pose_class_names = self.load_label(label_file_name)

    def load_label(self, label_path : str):
        with open(label_path, "r") as f:
            return [line.strip() for _, line in enumerate(f.readlines())]

    def classtify(self, keypoint_with_scores : np.ndarray):
        keypoint_with_scores = np.squeeze(keypoint_with_scores)
        kplist = []
        for kp in keypoint_with_scores:
            ky, kx, ks = kp
            kplist.append(kx)
            kplist.append(ky)
            kplist.append(ks)
        kplist = np.expand_dims(kplist, axis=0)
        self.interperter.set_tensor(self.input_detail, kplist)
        self.interperter.invoke()

        output = self.interperter.get_tensor(self.output_detail)
        output = np.squeeze(output, axis=0)


        return self.pose_class_names, output