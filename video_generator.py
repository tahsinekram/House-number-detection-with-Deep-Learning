import cv2
import os
from recognize import detect_and_classify

PATH = "./Data/"


def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            yield frame
        else:
            break
    video.release()
    yield None


def video_processor(video_name, fps):
    video = os.path.join(PATH, video_name)
    image_gen = video_frame_generator(video)

    image = image_gen.__next__()
    bh, bw, _ = image.shape
    out_path = "{}/OUT_VIDEO.mp4".format(PATH)
    video_out = mp4_video_writer(out_path, (bw, bh), fps)

    frame_num = 1
    while image is not None:
        print("Processing frame {}".format(frame_num))
        frame_num += 1

        # try:
        out_image = detect_and_classify(image)


        video_out.write(out_image)

        image = image_gen.__next__()

    video_out.release()

def video_processor_test(video_name, fps):
    video = os.path.join(PATH, video_name)
    image_gen = video_frame_generator(video)

    image = image_gen.__next__()

    while image is not None:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("few",image)
        bh, bw, _ = image.shape
        print (image.shape)
        cv2.waitKey(0)
        image = image_gen.__next__()


def mp4_video_writer(filename, frame_size, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


video_processor("input.mp4", 30)
