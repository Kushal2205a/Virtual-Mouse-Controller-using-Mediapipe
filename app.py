import csv, copy, argparse, itertools, pyautogui, screeninfo, time
from collections import Counter, deque
import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier

from screeninfo import get_monitors



pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


gaze_enabled = False
gaze_tx = None
gaze_ty = None
gaze_prev_x = 0.0
gaze_prev_y = 0.0
gaze_ts = 0.0


gaze_invert_y = True      
gaze_gain_x = 1.6         
gaze_gain_y = 1.8       
gaze_center_x = 0.50      
gaze_center_y = 0.50
gaze_last_rx = 0.50       
gaze_last_ry = 0.50


try:
    import ctypes
    
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

def get_virtual_screen():
    mons = get_monitors()
    minx = min(m.x for m in mons);              miny = min(m.y for m in mons)
    maxx = max(m.x + m.width for m in mons);    maxy = max(m.y + m.height for m in mons)
    return minx, miny, maxx - minx, maxy - miny

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    global gaze_enabled, gaze_tx, gaze_ty, gaze_prev_x, gaze_prev_y, gaze_ts, gaze_invert_y, gaze_gain_x, gaze_gain_y, gaze_center_x, gaze_center_y, gaze_last_rx, gaze_last_ry

   
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_complexity=0,
    )
    
    
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        refine_landmarks=True,      
        max_num_faces=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    keypoint_classifier = KeyPointClassifier()

    
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in reader if len(row) > 0]

    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    
    history_length = 16
    point_history = deque(maxlen=history_length)

    
    finger_gesture_history = deque(maxlen=history_length)

   
    mode = 0

    while True:
        fps = cvFpsCalc.get()

       
        key = cv.waitKey(10)
        if key == 27:  
            break
        number, mode = select_mode(key, mode)
        
       
        if key == ord('g'):
            global gaze_enabled
            gaze_enabled = not gaze_enabled
            print(f"[Gaze] {'ENABLED' if gaze_enabled else 'disabled'}")
            
        if key == ord('c'): 
            gaze_center_x, gaze_center_y = gaze_last_rx, gaze_last_ry
            print(f"[Gaze] center set to ({gaze_center_x:.2f},{ gaze_center_y:.2f})")
        if key == ord('y'):  
            gaze_invert_y = not gaze_invert_y
            print(f"[Gaze] invert_y = {gaze_invert_y}")
        if key == ord(']'):  
            gaze_gain_x *= 1.10; gaze_gain_y *= 1.10
            print(f"[Gaze] gain -> x:{gaze_gain_x:.2f} y:{gaze_gain_y:.2f}")

        if key == ord('['): 
            gaze_gain_x /= 1.10; gaze_gain_y /= 1.10
            print(f"[Gaze] gain -> x:{gaze_gain_x:.2f} y:{gaze_gain_y:.2f}")

        
        

        
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1) 
        debug_image = copy.deepcopy(image)
        frame_w = debug_image.shape[1]
        frame_h = debug_image.shape[0]
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        
        
       
        face_res = face_mesh.process(image)

        def _avg_xy(landmarks, idxs):
            xs, ys = [], []
            for i in idxs:
                xs.append(landmarks[i].x)
                ys.append(landmarks[i].y)
            return (sum(xs) / len(xs), sum(ys) / len(ys))

        def _eye_ratios(face_lms):
            
           
            L_OUT, L_IN = 33, 133
            R_IN, R_OUT = 362, 263  
            
            L_TOP, L_BOT = 159, 145
            R_TOP, R_BOT = 386, 374
            
            L_IRIS = [468, 469, 470, 471, 472]
            R_IRIS = [473, 474, 475, 476, 477]

            
            lx_out, ly_out = face_lms[L_OUT].x, face_lms[L_OUT].y
            lx_in,  ly_in  = face_lms[L_IN].x,  face_lms[L_IN].y
            lx_top, ly_top = face_lms[L_TOP].x, face_lms[L_TOP].y
            lx_bot, ly_bot = face_lms[L_BOT].x, face_lms[L_BOT].y
            li_x, li_y = _avg_xy(face_lms, L_IRIS)

            left_w  = max(1e-6, abs(lx_in - lx_out))
            left_h  = max(1e-6, abs(ly_bot - ly_top))
            left_rx = (li_x - min(lx_in, lx_out)) / left_w
            left_ry = (li_y - min(ly_top, ly_bot)) / left_h

            
            rx_in,  ry_in  = face_lms[R_IN].x,  face_lms[R_IN].y
            rx_out, ry_out = face_lms[R_OUT].x, face_lms[R_OUT].y
            rx_top, ry_top = face_lms[R_TOP].x, face_lms[R_TOP].y
            rx_bot, ry_bot = face_lms[R_BOT].x, face_lms[R_BOT].y
            ri_x, ri_y = _avg_xy(face_lms, R_IRIS)

            right_w  = max(1e-6, abs(rx_in - rx_out))
            right_h  = max(1e-6, abs(ry_bot - ry_top))
            right_rx = (ri_x - min(rx_in, rx_out)) / right_w
            right_ry = (ri_y - min(ry_top, ry_bot)) / right_h

            
            rx = float(np.clip((left_rx + right_rx) * 0.5, 0.0, 1.0))
            ry = float(np.clip((left_ry + right_ry) * 0.5, 0.0, 1.0))
            return rx, ry

        if face_res.multi_face_landmarks:
            
            face_lms = face_res.multi_face_landmarks[0].landmark
            rx, ry = _eye_ratios(face_lms)

           
            gaze_last_rx, gaze_last_ry = rx, ry

           
            rx_adj = (rx - gaze_center_x) * gaze_gain_x + 0.5
            ry_adj = (ry - gaze_center_y) * gaze_gain_y + 0.5

           
            if gaze_invert_y:
                ry_adj = 1.0 - ry_adj

           
            rx_adj = float(np.clip(rx_adj, 0.0, 1.0))
            ry_adj = float(np.clip(ry_adj, 0.0, 1.0))

           
            screen_width, screen_height = pyautogui.size()
            gx = rx_adj * (screen_width  - 1)
            gy = ry_adj * (screen_height - 1)

            a = 0.35  
            global gaze_tx, gaze_ty, gaze_prev_x, gaze_prev_y, gaze_ts
            gaze_prev_x = gaze_prev_x + (gx - gaze_prev_x) * a
            gaze_prev_y = gaze_prev_y + (gy - gaze_prev_y) * a

            gaze_tx, gaze_ty = gaze_prev_x, gaze_prev_y
            gaze_ts = time.time()


        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                gesture_name = keypoint_classifier_labels[hand_sign_id]
                
                
                def thumb_index_ratio(landmarks, brect):
                    
                    x1, y1 = landmarks[4]
                    x2, y2 = landmarks[8]
                    d = float(np.hypot(x1 - x2, y1 - y2))
                    hand_size = float(max(1, max(brect[2] - brect[0], brect[3] - brect[1])))
                    return d / hand_size

                ratio = thumb_index_ratio(landmark_list, brect)

               
                if gesture_name == "OK" and ratio > 0.32:
                    gesture_name = "Open"

                
                elif gesture_name == "Open" and ratio < 0.18:
                    gesture_name = "OK"


                
                screen_width, screen_height = pyautogui.size()
                smooth_factor = 0.25
            

                margin_x, margin_y = 0, 0
                global prev_x, prev_y,last_gesture,last_action_time 
                try:
                    prev_x
                except NameError:
                    prev_x, prev_y = pyautogui.position()
                    last_gesture = None
                    last_action_time = 0

                now = time.time()

            
                if last_gesture == "Pointer" and gesture_name != "Pointer":
                    try:
                        pyautogui.mouseUp(button="left")
                    except Exception:
                        pass

                def move_cursor_from_index():
                    """Map index fingertip to screen with smoothing, then move."""
                    global prev_x, prev_y, gaze_tx, gaze_ty,gaze_ts
                   
                    x, y = landmark_list[8]

                   
                    margin_x, margin_y = 0, 0

                   
                    x = np.clip(x, margin_x, frame_w - 1 - margin_x)
                    y = np.clip(y, margin_y, frame_h - 1 - margin_y)

                    
                    screen_x = np.interp(x, [margin_x, frame_w - margin_x], [0, screen_width])
                    screen_y = np.interp(y, [margin_y, frame_h - margin_y], [0, screen_height])
                    
                    if gaze_enabled and (gaze_tx is not None) and (time.time() - gaze_ts) < 0.2:
                        lam = 0.35 
                        screen_x = (1 - lam) * screen_x + lam * gaze_tx
                        screen_y = (1 - lam) * screen_y + lam * gaze_ty

                    
                    prev_x = prev_x + (screen_x - prev_x) * smooth_factor
                    prev_y = prev_y + (screen_y - prev_y) * smooth_factor

                    
                    prev_x = max(0, min(screen_width - 1, prev_x))
                    prev_y = max(0, min(screen_height - 1, prev_y))
                    pyautogui.moveTo(prev_x, prev_y)

                
               
                try:
                    pose_buf
                except NameError:
                    pose_buf = deque(maxlen=7)     
                    stable_pose = None
                    click_guard_until = 0.0
                    suppress_left_until = 0.0
                    suppress_right_until = 0.0
                    pointer_down = False
                    movement_hold_until = 0.0
                    anchor_x, anchor_y = prev_x, prev_y
                    last_open_x, last_open_y = prev_x, prev_y
                    action_guard_until = 0.0     


               
                pose_buf.append(gesture_name)
                candidate = Counter(pose_buf).most_common(1)[0][0]

                old_stable = stable_pose
                if stable_pose is None:
                    stable_pose = candidate
                elif candidate != stable_pose and pose_buf.count(candidate) >= 4: 
                    stable_pose = candidate

                entered = (stable_pose != old_stable)
                now = time.time()

               
                if gesture_name == "Close":
                    suppress_right_until = max(suppress_right_until, now + 0.60)

                
                if (gesture_name in ("Close", "OK")) or (candidate in ("Close", "OK")):
                    movement_hold_until = max(movement_hold_until, now + 0.25)

                
                if pointer_down and stable_pose != "Pointer":
                    try:
                        pyautogui.mouseUp(button="left")
                    except Exception:
                        pass
                    pointer_down = False

                def maybe_move():
                    """Move only if we're not in a freeze window."""
                    if time.time() >= movement_hold_until:
                        move_cursor_from_index()

               

                if stable_pose == "Open":
                    maybe_move()
                    
                    last_open_x, last_open_y = prev_x, prev_y

                elif stable_pose == "Pointer":
                    if entered and now > suppress_left_until and now > click_guard_until:
                        pyautogui.mouseDown(button="left")
                        pointer_down = True
                        movement_hold_until = max(movement_hold_until, now + 0.08)
                    maybe_move()

                elif stable_pose == "Close":
                    
                    if entered and (now > click_guard_until) and (now > suppress_left_until):
                        pyautogui.moveTo(last_open_x, last_open_y)
                        pyautogui.click(button="left")
                        click_guard_until     = now + 0.30
                        suppress_right_until  = now + 0.60  
                        movement_hold_until   = max(movement_hold_until, now + 0.25)


                elif stable_pose == "OK":
                    
                    if entered and (now > click_guard_until) and (now > suppress_right_until):
                        pyautogui.moveTo(last_open_x, last_open_y)  
                        pyautogui.click(button="right")
                        click_guard_until     = now + 0.70
                        suppress_left_until   = now + 0.60
                        movement_hold_until   = max(movement_hold_until, now + 0.45)
                        
               
                elif stable_pose in ("Peace",):
                    if entered and now > action_guard_until:
                        pyautogui.doubleClick()
                        action_guard_until = now + 0.40
                        movement_hold_until = max(movement_hold_until, now + 0.10)

                
                elif stable_pose in ("Thumbsup", "Thumbs up"):
                    if entered and now > action_guard_until:
                        try:
                            pyautogui.press('playpause')  
                        except Exception:
                            pyautogui.press('space')      
                        action_guard_until = now + 0.50
                        movement_hold_until = max(movement_hold_until, now + 0.05)

                # Thumbsdown -> close window
                elif stable_pose in ("Thumbsdown", "Thumbs down"):
                    if entered and now > action_guard_until:
                        pyautogui.hotkey('alt', 'f4')
                        action_guard_until = now + 1.00    
                        movement_hold_until = max(movement_hold_until, now + 0.20)

                # Call -> TAB SWITCH (Ctrl+Tab)
                elif isinstance(stable_pose, str) and stable_pose.lower() == "call":
                    if entered and now > action_guard_until:
                        pyautogui.keyDown('alt')
                        time.sleep(0.15)
                        pyautogui.press('tab')
                        time.sleep(0.15)
                        pyautogui.keyDown('alt')

                elif stable_pose in ("threefingersup", "Threefingersup", "Three fingers up"):
                    if entered and now > action_guard_until:
                        pyautogui.press('volumeup')
                        action_guard_until = now + 0.20

                # Rock -> volume down
                elif stable_pose in ("rock", "Rock"):
                    if entered and now > action_guard_until:
                        pyautogui.press('volumedown')
                        action_guard_until = now + 0.20


                else:
                    pass

                last_gesture = stable_pose
                finger_gesture_id = 0
                
    
                    

               
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        cv.namedWindow('Hand Gesture Recognition', cv.WINDOW_NORMAL)
        cv.setWindowProperty('Hand Gesture Recognition', cv.WND_PROP_TOPMOST, 1)


       
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history



sample_counts = {i: 0 for i in range(10)}

def logging_csv(number, mode, landmark_list, point_history_list):
    global sample_counts
    if mode == 0:
        return

    if 0 <= number <= 9:
        if mode == 1:
            csv_path = 'model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        elif mode == 2:
            csv_path = 'model/point_history_classifier/point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])

        # Increment count and print feedback
        sample_counts[number] += 1
        print(f"Class {number} sample #{sample_counts[number]}")


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4: 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20: 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    """if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)"""

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
