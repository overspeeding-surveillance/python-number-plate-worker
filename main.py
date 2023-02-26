import cv2
import torch
from custom_utils.capture import capture_plate
import pika
import sys
import os
import uuid

SECOND_PYTHON_QUEUE = "SECOND_PYTHON_QUEUE"
THIRD_PYTHON_QUEUE = "THIRD_PYTHON_QUEUE"

model = torch.hub.load("ultralytics/yolov5", "custom",
                       path="best.pt", force_reload=True)


def main():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue=SECOND_PYTHON_QUEUE)

    def callback(ch, method, properties, body):
        print(" [x] Received %r" % body)
        vehicle_img_path = "../vehicles/" + str(body.decode())
        print(vehicle_img_path)

        if not os.path.exists(vehicle_img_path):
            return
        results = model(vehicle_img_path)
        [readable_result] = results.pandas().xyxy

        img = cv2.imread(vehicle_img_path, 0)

        for i in range(0, len(readable_result)):
            xmin = int(readable_result['xmin'][i])
            ymin = int(readable_result['ymin'][i])
            xmax = int(readable_result['xmax'][i])
            ymax = int(readable_result['ymax'][i])
            filename = str(uuid.uuid4()) + ".jpg"
            capture_plate(img, xmin, ymin, xmax, ymax, filename)

    channel.basic_consume(
        queue=SECOND_PYTHON_QUEUE, on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
