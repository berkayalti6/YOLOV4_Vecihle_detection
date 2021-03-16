import cv2
import numpy as np



MODEL = "C:/Users/user/Desktop/pythonProject/YOLOV4-DENEME/data/yolov4.weights"
CFG = "C:/Users/user/Desktop/pythonProject/YOLOV4-DENEME/data/yolov4.cfg"

net = cv2.dnn.readNetFromDarknet(CFG, MODEL)

classes = []
with open("C:/Users/user/Desktop/pythonProject/YOLOV4-DENEME/data/coco.names", "r") as f:
    classes = f.read().splitlines()
print(f"Total: {len(classes)} classes and they are: \n{classes}")

myClass = ["car", "bus", "truck", "motorbike", "bicycle"]
print(myClass)


output_layers_names = net.getUnconnectedOutLayersNames()
type(output_layers_names), len(output_layers_names)
all_layers = net.getLayerNames()
cnn_layers = [layer for layer in all_layers if "conv" in layer]


def preprocess(frame):
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)
    layerOutputs = net.forward(output_layers_names)

    return layerOutputs


def detect_objects(layerOutputs, height, width):
    boxes, confidences, class_ids = [], [], []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids


def lane_divider(frame):
    cv2.line(frame, (350, 900), (1010, 900), (0, 255, 255), 3)
    cv2.line(frame, (1100, 800), (1650, 800), (0, 255, 255), 3)
    #cv2.line(frame, (50, 300), (600, 300), (0, 255, 255), 3)

countDown= 0
countUp = 0
c = 0
cl = 0
t = 0
tl = 0
while True:
    cap = cv2.VideoCapture('C:/Users/user/Desktop/pythonProject/YOLOV4-DENEME/examples/Highway.mp4')

    while (cap.isOpened()):
        ret, frame = cap.read()
        height, width = frame.shape[0:2]



        layerOutputs = preprocess(frame)
        boxes, confidences, class_ids = detect_objects(layerOutputs, height, width)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        lane_divider(frame)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]

                if label in myClass:
                    rect = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    center = cv2.circle(frame, (int(x+w/2),int(y+h/2)), radius=2, color=(0, 0, 255), thickness=2)
                    #rect[(y - 15):(y + 5), x:(x + w)] = color
                    cv2.putText(frame, label + " " + confidence, (x, y), font, 1.3, (0, 0, 0), 2)

                    # logics for vehicle counting
                    bikeC1y = int(y + h / 2)
                    linC1y = 900
                    bikeC2y = int(y + h / 2)
                    linC2y = 800

                    if (bikeC1y < linC1y + 4 and bikeC1y > linC1y - 4 and x <= 1010):
                        countDown = countDown + 1
                        cv2.line(frame, (350, 900), (1010, 900), (0, 0, 255), 3)
                        if label == 'car':
                            c += 1
                        if label == 'truck':
                            t += 1

                    if (bikeC2y < linC2y + 5 and bikeC2y > linC2y - 4 and x >= 1100):
                        countUp = countUp + 1
                        cv2.line(frame, (1100, 800), (1650, 800), (0, 0, 255), 3)
                        if label == 'car':
                            cl += 1
                        if label == 'truck':
                            tl += 1

                    cv2.putText(frame,"TOTAL LEFT: " + str(countDown), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                    cv2.putText(frame,"TOTAL RIGHT: " + str(countUp), (1200, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                    cv2.putText(frame, "Car: " + str(c), (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0),3)
                    cv2.putText(frame, "Truck: " + str(t), (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                    cv2.putText(frame, "Car: " + str(cl), (1200, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                    cv2.putText(frame, "Truck: " + str(tl), (1200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        frame = cv2.resize(frame, (1300, 700))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()