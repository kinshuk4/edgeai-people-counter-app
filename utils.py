import os
import mimetypes
import cv2
'''
    Returns if file is single image and the stream
'''
def get_file_type(input_file):
    if input_file.upper() == "CAM":
        return False, 0
    abs_path = os.path.abspath(input_file)
    mime_type = mimetypes.guess_type(abs_path)
    if "image" in mime_type[0]:
        return True, input_file
    elif "video" in mime_type[0]:
        return False, input_file
    else:
        print("Given format for {} is not supported".format(input_file))
        exit(1)

def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to height and width
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start
    '''
    image = cv2.resize(input_image, (width, height)) # note width first
    image = image.transpose((2,0,1)) # We first put our 3rd channel and then 0 and 1 - corresponding to width and height
    image = image.reshape(1, *image.shape) # note height first

    return image


def draw_boxes(image, result, width, height, threshold):
    '''
    Draw bounding boxes onto the frame.
    '''
    count = 0
    thickness = 1 # line thickness
    color = (255, 0, 255)
    for box in result[0][0]:
        confidence = box[2]
        if confidence >= threshold:
            count += 1
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

    return image, count