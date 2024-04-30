import math


def compute_midpoint(p1,p2):
    return [int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)]


def get_ear_values(detection_result):
    face_landmarks_list = detection_result.face_landmarks

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Right Eye
        #print(annotated_image.shape)
        pointA = [face_landmarks[33].x, face_landmarks[33].y]
        pointB = [face_landmarks[133].x, face_landmarks[133].y]
        pointC = [face_landmarks[145].x, face_landmarks[145].y]
        pointD = [face_landmarks[159].x, face_landmarks[159].y]
        pointE = [face_landmarks[158].x, face_landmarks[158].y]
        pointF = compute_midpoint(pointD, pointE)
        #print(pointA)

        ear_right_eye = math.dist(pointC, pointF) / math.dist(pointA, pointB)
        print(pointC, pointF, math.dist(pointC, pointF))

        # Left Eye
        pointA = [face_landmarks[362].x, face_landmarks[362].y]
        pointB = [face_landmarks[263].x, face_landmarks[263].y]
        pointC = [face_landmarks[374].x, face_landmarks[374].y]
        pointD = [face_landmarks[385].x, face_landmarks[385].y]
        pointE = [face_landmarks[386].x, face_landmarks[386].y]
        pointF = compute_midpoint(pointD, pointE)
        # print(pointA)

        ear_left_eye = math.dist(pointC, pointF) / math.dist(pointA, pointB)

        return ear_right_eye, ear_left_eye