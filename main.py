import numpy
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow
from gtts import gTTS
import cv2
sys.path.append("..")
from src.utils import label_map_util
from src.utils import visualization_utils

video_stream = cv2.VideoCapture(0)
sys.path.append("..")

prefix = "models/"
try:
    os.makedirs(prefix)
except:
    pass

model_name = 'ssd_inception_v2_coco_2017_11_17'
model = prefix + model_name + '/frozen_inference_graph.pb'
if not os.path.isfile(model):
    site = 'http://download.tensorflow.org/models/object_detection/'
    file_extension = '.tar.gz'
    archive_name = model_name + file_extension
    opener = urllib.request.URLopener()
    opener.retrieve(site + archive_name, archive_name)
    tar = tarfile.open(archive_name)
    for file in tar.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar.extract(file, os.getcwd() + "/" + prefix)

    os.remove(archive_name)

detection_graph = tensorflow.Graph()
with detection_graph.as_default():
    graph_def = tensorflow.GraphDef()
    with tensorflow.gfile.GFile(model, 'rb') as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
        tensorflow.import_graph_def(graph_def, name='')

PATH_TO_LABELS = os.path.join('src', 'data', 'mscoco_label_map.pbtxt')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=90,     # Change if using a different model than default
    use_display_name=True
)
category_index = label_map_util.create_category_index(categories)


def speak(words):
    tts = gTTS(text=words, lang='en')
    tts.save('tmp.mp3')
    os.system("mpg123 -q tmp.mp3")
    os.remove("tmp.mp3")


objects = []

with detection_graph.as_default():
    with tensorflow.Session(graph=detection_graph) as session:
        n = 0
        while True:
            ret, image_np = video_stream.read()
            image_np_expanded = numpy.expand_dims(image_np, axis=0)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0'
            )

            (boxes, scores, classes, num_detections) = session.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded}
            )

            for i in range(0, len(scores[0])):
                n = n + 1
                if scores[0][i] < 0.8:
                    break

                obj = category_index.get(classes[0][i]).get('name')
                if obj == 'airplane':
                    continue

                if obj not in objects:
                    objects.append(obj)
                    print(obj)
                    speak(obj)


                if n > 2:
                    objects = []
                    n = 0
