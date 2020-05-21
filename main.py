"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2

import logging
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
import utils

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# logging
logging.basicConfig(filename="./people-counter.log", level=logging.INFO)


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    return parser


def connect_mqtt():
    ### DONE: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### DONE: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    logging.debug("Loaded the model. Shape: {}".format(net_input_shape))
    ### DONE: Handle the input stream ###
    is_single_image, stream = utils.get_file_type(args.input)
    logging.debug("Got the stream: {}".format(stream))

    camera = cv2.VideoCapture(stream)

    stream_width = int(camera.get(3))
    stream_height = int(camera.get(4))
    if not camera.isOpened():
        logging.error("Error opening video stream {}".format(args.input))
        exit(1)

    last_count = 0
    total_count = 0

    frames_counter = 0  # Number of frames people on screen
    contiguous_frame_counter = 0  # Number of contiguous frames with same number of people on it
    fps = int(camera.get(cv2.CAP_PROP_FPS))  # Frames per sec, was 10
    logging.debug("FPS: {}".format(fps))
    prev_count = 0
    contiguous_count_threshold = 5  # half a second as threshold as FPS = 10
    ### DONE: Loop until stream is over ###
    while camera.isOpened():
        ### DONE: Read from the video capture ###
        more_image_flag, input_frame = camera.read()
        if not more_image_flag:
            break
        key_pressed = cv2.waitKey(60)
        if key_pressed == 27:
            break
        ### DONE: Pre-process the image as needed ###
        preprocessed_frame = utils.preprocessing(input_frame, net_input_shape[2], net_input_shape[3])
        ### DONE: Start asynchronous inference for specified request ###
        inference_start_time = time.time()
        infer_network.exec_net(preprocessed_frame)

        ### DONE: Wait for the result ###
        if infer_network.wait() == 0:
            ### DONE: Get the results of the inference request # ##
            inference_duration = time.time() - inference_start_time

            result = infer_network.get_output()
            ### DONE: Extract any desired stats from the results ###
            output_frame, current_count = utils.draw_boxes(input_frame, result, stream_width,
                                                           stream_height, prob_threshold)

            ### DONE: Calculate and send relevant information on ###
            inference_message = "Inference time: {:.3f}ms".format(inference_duration * 1000)
            cv2.putText(output_frame, inference_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            frames_counter += 1
            logging.debug(
                "prev:{} , current:{} , frames:{}, cont_frames:{}, last:{} , total:{}".format(prev_count, current_count,
                                                                                              frames_counter,
                                                                                              contiguous_frame_counter,
                                                                                              last_count, total_count))
            if current_count != prev_count:
                prev_count = current_count
                contiguous_frame_counter = 0
            else:
                contiguous_frame_counter += 1
                if contiguous_frame_counter >= contiguous_count_threshold:
                    if current_count > last_count:
                        total_count += (current_count - last_count)
                        frames_counter = 0  # reset for new people in frame
                        client.publish("person", json.dumps({"total": total_count, "count": current_count}))
                    elif current_count < last_count:
                        duration = int(frames_counter / fps)
                        client.publish("person/duration", json.dumps({"duration": duration}))
                    last_count = current_count
            ### DONE: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(output_frame)
            sys.stdout.flush()
            ### DONE: Write an output image if `single_image_mode` ###
            if is_single_image:
                outputFileName = "out_" + os.path.basename(args.input)
                cv2.imwrite(outputFileName, output_frame)
    camera.release()
    cv2.destroyAllWindows()
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
