"""
/***************************************************************************
  my_nanny_cam.py
  
  A script that analyzes a selected fragment (rectangle_to_guard) of an image from
  an mp4 file (one can increase the speed of player), local camera (USB camera)
  or remote camera (IP camera) and saves the image in a video file (mp4)
  in case of changes.
  The settings are in the setup selection of this script.
  If you set the template to show - it will be possible to set a mask.
  This script uses opencv and numpy libraries.
  --------------------------------------------------------------------------
  Copyright: (C) 2019 by Piotr Michałowski, Olsztyn, woj. W-M, Poland
  Email: piotrm35@hotmail.com
  Python 3.x.x
  Date : 16.04.2021
/***************************************************************************
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as published
 * by the Free Software Foundation.
 *
 ***************************************************************************/
"""

SCRIPT_NAME = 'my_nanny_cam v. 2.1'

#--------------------------------------------------------------------------------------------------------------------------------------
# installation of libraries (Ubuntu):


# sudo pip3 install numpy
# sudo apt-get install python3-opencv
# python3 -c "import cv2; my_print(cv2.__version__)"    # my result: 'v. 4.2.0+dfsg-5'


#======================================================================================================================================
# setup:


VIDEO_RESOLUTION = {'QVGA': (320, 240), 'VGA': (640, 480), 'HD_ready': (1280, 720), 'Full_HD': (1920, 1080)}

PREWIEV_WINDOW_SIZE = VIDEO_RESOLUTION['VGA']
SHOW_TEMPLATE = True
SHOW_ROI = False
SHOW_MOVE_DETECT = False
VERBOSE_TEST_PRINT = False
VIDEO_INPUT_FOLDER = 'video_input'
VIDEO_OUTPUT_FOLDER = 'video_output'

BINARY_THRESHOLD_INT = 20                   # {0..255} used in cv2.threshold of roi -> move_detect
VIDEO_RECORD_THRESHOLD_INT = 50 * 255       # N * 255 -> N: the number of pixels that have changed in rectangle_to_guard (=roi)
VIDEO_RECORDING_POST_TIME_SEC = 1

VIDEO_SOURCE = 'local_cam'                  # VIDEO_SOURCE = 'mp4_file' or 'local_cam' or 'remote_cam'
INPUT_MP4_FILE_NAME = 'example.mp4'
MP4_PLAYER_FPS = 90                         # one can increase the speed of player; if MP4_PLAYER_FPS = None -> used FPS from mp4 file;
LOCAL_CAM_IDX = 0
REMOTE_CAM_URL = 'rtsp://172.19.0.10/video.mp4'


#--------------------------------------------------------------------------------------------------------------------------------------
#globals:


import os, time
import cv2
import numpy as np
from threading import Timer


cap = None
if VIDEO_SOURCE == 'mp4_file':
    cap = cv2.VideoCapture(os.path.join(VIDEO_INPUT_FOLDER, INPUT_MP4_FILE_NAME))
elif VIDEO_SOURCE == 'local_cam':
    cap = cv2.VideoCapture(LOCAL_CAM_IDX)
elif VIDEO_SOURCE == 'remote_cam':
    cap = cv2.VideoCapture(REMOTE_CAM_URL)

current_camera_frame = None
current_camera_frame_TMP = None
rectangle_to_guard  = None
template_and_mask_in_one = None
template_img = None
rec_mouse_start_pos = None
video_pause = False
user_trackbar_change = True
mask_mouse_prev_pos = None
mask_mouse_down_Timer = None
mp4_fourcc = cv2.VideoWriter_fourcc(*'MP4V')
mp4_out = None
mp4_out_timeout = None
mp4_out_write = False
VIDEO_HIGHT = None
VIDEO_WIDTH = None
FPS = None
prev_roi = None
prev_roi_diff = None


#--------------------------------------------------------------------------------------------------------------------------------------
# GUI functions:


def on_trackbar_change(x):
    global user_trackbar_change
    if user_trackbar_change:
        new_start_frame_no = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * x / 100)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_start_frame_no)
    user_trackbar_change = True


def on_mouse_move_draw_rectangle(event, x, y, flags, user_data):
    global rec_mouse_start_pos
    global rectangle_to_guard 
    global template_and_mask_in_one
    global template_img
    global video_pause
    global prev_roi
    global prev_roi_diff
    if event == cv2.EVENT_LBUTTONDOWN:
        rectangle_to_guard  = None
        template_and_mask_in_one = None
        prev_roi = None
        prev_roi_diff = None
        rec_mouse_start_pos = (x, y)
        video_pause = True
        destroy_aux_windows()
    elif event == cv2.EVENT_LBUTTONUP:
        video_pause = False
        rec_mouse_start_pos = None
        destroy_aux_windows()
        if rectangle_to_guard is not None:
            rectangle_to_guard_HEIGHT = rectangle_to_guard[3] - rectangle_to_guard[1]
            rectangle_to_guard_WIDTH = rectangle_to_guard[2] - rectangle_to_guard[0]
            if rectangle_to_guard_HEIGHT > 10 and rectangle_to_guard_WIDTH > 10:
                template_and_mask_in_one = np.ones((rectangle_to_guard_HEIGHT, rectangle_to_guard_WIDTH), dtype='uint8')
                if SHOW_TEMPLATE:
                    template_img = get_img_part(0)
                    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('tpl', template_img)
                    cv2.setMouseCallback('tpl', on_mouse_move_draw_mask)
    elif flags & cv2.EVENT_FLAG_LBUTTON:
        if rec_mouse_start_pos is not None:
            min_pos = (min(rec_mouse_start_pos[0], x), min(rec_mouse_start_pos[1], y))
            max_pos = (max(rec_mouse_start_pos[0], x), max(rec_mouse_start_pos[1], y))
            rectangle_to_guard  = (min_pos[0], min_pos[1], max_pos[0], max_pos[1])
    elif event == cv2.EVENT_RBUTTONDOWN:
        rec_mouse_start_pos = None
        rectangle_to_guard  = None
        template_and_mask_in_one = None
        video_pause = False
        prev_roi = None
        prev_roi_diff = None
        destroy_aux_windows()


def destroy_aux_windows():
    try:
        cv2.destroyWindow('tpl')
    except:
        pass
    try:
        cv2.destroyWindow('roi')
    except:
        pass
    try:
        cv2.destroyWindow('move_detect')
    except:
        pass


def set_mask_mouse_prev_pos_to_None():
    global mask_mouse_prev_pos
    mask_mouse_prev_pos = None

    
def on_mouse_move_draw_mask(event, x, y, flags, user_data):
    global template_and_mask_in_one
    global mask_mouse_prev_pos
    global mask_mouse_down_Timer
    if event == cv2.EVENT_LBUTTONDOWN:
        if template_and_mask_in_one is not None:
            mask_mouse_prev_pos = (x, y)
            if mask_mouse_down_Timer:
                mask_mouse_down_Timer.cancel()
            mask_mouse_down_Timer = Timer(2, set_mask_mouse_prev_pos_to_None)
            mask_mouse_down_Timer.start()
    elif event == cv2.EVENT_LBUTTONUP:
        if template_and_mask_in_one is not None and mask_mouse_prev_pos is not None:
            mask_mouse_prev_pos = None
            if mask_mouse_down_Timer:
                mask_mouse_down_Timer.cancel()
                mask_mouse_down_Timer = None
    elif flags & cv2.EVENT_FLAG_LBUTTON:
        if template_and_mask_in_one is not None and mask_mouse_prev_pos is not None:
            cv2.line(template_and_mask_in_one, mask_mouse_prev_pos, (x, y), 0, 10)
            masked_template = cv2.bitwise_and(template_img, template_img, mask = template_and_mask_in_one)
            cv2.imshow('tpl', masked_template)
            if mask_mouse_down_Timer:
                mask_mouse_down_Timer.cancel()
            mask_mouse_down_Timer = Timer(2, set_mask_mouse_prev_pos_to_None)
            mask_mouse_down_Timer.start()
            mask_mouse_prev_pos = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        template_and_mask_in_one = np.ones((abs(rectangle_to_guard[3] - rectangle_to_guard[1]), abs(rectangle_to_guard[2] - rectangle_to_guard[0])), dtype='uint8')
        if SHOW_TEMPLATE:
            cv2.imshow('tpl', template_img)


def draw_rectangle_to_guard ():
    global current_camera_frame_TMP
    if rectangle_to_guard  is not None:
        cv2.rectangle(current_camera_frame_TMP, (rectangle_to_guard [0], rectangle_to_guard [1]), (rectangle_to_guard [2], rectangle_to_guard [3]), (0, 255, 255), 1)


#--------------------------------------------------------------------------------------------------------------------------------------
# work functions:


def get_img_part(margin):
    if(current_camera_frame is not None and rectangle_to_guard  is not None):
        roi_x_1 = rectangle_to_guard [0] - margin
        roi_y_1 = rectangle_to_guard [1] - margin
        if roi_x_1 < 0:
            roi_x_1 = 0
        if roi_y_1 < 0:
            roi_y_1 = 0
        current_camera_frame_HEIGHT, current_camera_frame_WIDTH = current_camera_frame.shape[:2]
        roi_x_2 = rectangle_to_guard [2] + margin
        roi_y_2 = rectangle_to_guard [3] + margin
        if roi_x_2 > current_camera_frame_WIDTH - 1:
            roi_x_2 = current_camera_frame_WIDTH - 1
        if roi_y_2 > current_camera_frame_HEIGHT - 1:
            roi_y_2 = current_camera_frame_HEIGHT - 1
        return current_camera_frame[roi_y_1:roi_y_2, roi_x_1:roi_x_2]
    else:
        return None


def video_recording_end():
    global mp4_out_write
    global mp4_out_timeout
    if mp4_out_write:
        my_print('video recording end\n')
        if mp4_out:
            mp4_out.release()
    mp4_out_write = False
    mp4_out_timeout = None


def match():
    global mp4_out
    global mp4_out_timeout
    global mp4_out_write
    global prev_roi
    global prev_roi_diff
    if mp4_out_timeout and time.time() - mp4_out_timeout > VIDEO_RECORDING_POST_TIME_SEC:
        video_recording_end()
    if template_and_mask_in_one is not None:
        roi = get_img_part(0)
        if roi is not None:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if SHOW_ROI:
                cv2.imshow('roi', roi)
            if  prev_roi is not None and prev_roi_diff is not None:
                roi_diff = cv2.absdiff(prev_roi, roi)
                move_detect = cv2.bitwise_and(prev_roi_diff, roi_diff)
                ret,move_detect = cv2.threshold(move_detect, BINARY_THRESHOLD_INT, 255, cv2.THRESH_BINARY)
                prev_roi = roi
                prev_roi_diff = roi_diff
                if SHOW_MOVE_DETECT:
                    cv2.imshow('move_detect', move_detect)
                match_result = cv2.matchTemplate(move_detect, template_and_mask_in_one, cv2.TM_CCORR)
                (my_min, my_max, min_loc, max_loc) = cv2.minMaxLoc(match_result)
                if not isinstance(my_max, float):
                    my_max = -1.0
                if(my_max > VIDEO_RECORD_THRESHOLD_INT):
                    if not mp4_out_write:
                        if VIDEO_SOURCE == 'mp4_file':
                            mp4_file_position_percent = '{:0.2f}'.format(100 * cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            output_file_name = os.path.join(VIDEO_OUTPUT_FOLDER, mp4_file_position_percent + '_PERCENT_' + INPUT_MP4_FILE_NAME)
                        else:
                            output_file_name = os.path.join(VIDEO_OUTPUT_FOLDER, time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime(time.time())) + '_video_output.mp4')
                        if mp4_out:
                            mp4_out.open(output_file_name, mp4_fourcc, FPS, (VIDEO_WIDTH, VIDEO_HIGHT))
                        else:
                            mp4_out = cv2.VideoWriter(output_file_name, mp4_fourcc, FPS, (VIDEO_WIDTH, VIDEO_HIGHT))
                        mp4_out_write = True
                        my_print('video recording begin to: ' + output_file_name)
                    mp4_out_timeout = time.time()
                    color = (0, 0, 255)
                    if VERBOSE_TEST_PRINT:
                        my_print('my_action: my_max = ' + str(int(my_max)) + ' > ' + str(VIDEO_RECORD_THRESHOLD_INT))
                else:
                    color = (0, 255, 0)
                    if VERBOSE_TEST_PRINT:
                        my_print('my_max = ' + str(int(my_max)))
                cv2.rectangle(current_camera_frame, (rectangle_to_guard [0], rectangle_to_guard [1]), (rectangle_to_guard [2], rectangle_to_guard [3]), color, 2)
            else:
                if  prev_roi is not None:
                    prev_roi_diff = cv2.absdiff(prev_roi, roi)
                prev_roi = roi
                

def my_print(tx):
    try:
        print(tx)
    except:
        pass


#--------------------------------------------------------------------------------------------------------------------------------------
# main script:


my_print('SCRIPT BEGIN')
my_print("press 'q' button to end the script")

if cap.isOpened():
    ret, current_camera_frame = cap.read()
    if ret == True:
        (VIDEO_HIGHT, VIDEO_WIDTH) = current_camera_frame.shape[:2]
        if VIDEO_SOURCE == 'mp4_file' and MP4_PLAYER_FPS is not None:
            FPS = MP4_PLAYER_FPS
        else:
            FPS = int(cap.get(cv2.CAP_PROP_FPS))
        loop_delay = int(1000 / FPS)
        PREWIEV_WINDOW_NAME = SCRIPT_NAME
        current_camera_frame = cv2.resize(current_camera_frame, PREWIEV_WINDOW_SIZE)
        cv2.namedWindow(PREWIEV_WINDOW_NAME, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(PREWIEV_WINDOW_NAME, on_mouse_move_draw_rectangle)
        if VIDEO_SOURCE == 'mp4_file':
            cv2.createTrackbar('video [%]:', PREWIEV_WINDOW_NAME, 0, 100, on_trackbar_change)
    else:
        my_print("cap.read() -> ret == False")
else:
    my_print("cap is not open.")

my_print('SCRIPT while loop begin')
fps_measure_start_time = time.time()
fps_measure_frames_count = 0
previous_fps_measure_frames_count = 0
while cap.isOpened():
    if not video_pause:
        ret, current_camera_frame = cap.read()
        if VIDEO_SOURCE == 'mp4_file':
            new_trackbar_value = int(100 * cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FRAME_COUNT))
            user_trackbar_change = False
            try:
                cv2.setTrackbarPos('video [%]:', PREWIEV_WINDOW_NAME, new_trackbar_value)
            except:
                pass
        if ret == True:
            if mp4_out_write:
                mp4_out.write(current_camera_frame)
            current_camera_frame = cv2.resize(current_camera_frame, PREWIEV_WINDOW_SIZE)
            match()
            cv2.imshow(PREWIEV_WINDOW_NAME, current_camera_frame)
        else:
            break
        fps_measure_frames_count += 1
        if time.time() - fps_measure_start_time >= 1:
            if fps_measure_frames_count != previous_fps_measure_frames_count:
                previous_fps_measure_frames_count = fps_measure_frames_count
                my_print('\nCURRENT FPS = ' + str(fps_measure_frames_count) + '\n')
            fps_diff = fps_measure_frames_count - FPS
            if fps_diff > 1:
                loop_delay += 1
            elif fps_diff < -1:
                loop_delay -= 1
            if loop_delay <= 0:
                loop_delay = 1
            fps_measure_start_time = time.time()
            fps_measure_frames_count = 0
    else:
        current_camera_frame_TMP = current_camera_frame.copy()
        draw_rectangle_to_guard ()
        cv2.imshow(PREWIEV_WINDOW_NAME,current_camera_frame_TMP)
    if cv2.waitKey(loop_delay) & 0xFF == ord('q'):
        break

my_print('SCRIPT ending')
cap.release()
cv2.destroyAllWindows()
video_recording_end()
my_print('SCRIPT END')

#======================================================================================================================================

