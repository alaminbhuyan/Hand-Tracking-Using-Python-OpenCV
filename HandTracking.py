import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
# set width and height
cap.set(3, 880)
cap.set(4, 640)
detector = HandDetector(maxHands=2, detectionCon=0.8)

while True:
    success, img = cap.read()
    if success:
        all_hands, img = detector.findHands(img=img, draw=True)
        # print(len(all_hands))
        if all_hands:
            # print(all_hands[0])
            # Take the first-hand information
            hand1 = all_hands[0]
            # List of 21 landmarks finger points
            landmark_list1 = hand1['lmList']
            # bounding box information like x, y, height, width
            bbox1 = hand1['bbox']
            # Center point
            center_point1 = hand1['center']
            hand_type1 = hand1['type']
            # print(bbox1)
            finger1 = detector.fingersUp(myHand=hand1)
            # length, info= detector.findDistance(p1=landmark_list1[8], p2=landmark_list1[8])

            if len(all_hands) == 2:
                # Take the first-hand information
                hand2 = all_hands[1]
                # List of 21 landmarks finger points
                landmark_list2 = hand2['lmList']
                # bounding box information like x, y, height, width
                bbox2 = hand2['bbox']
                # Center point
                center_point2 = hand2['center']
                hand_type2 = hand2['type']
                finger2 = detector.fingersUp(myHand=hand2)
                print(finger1, finger2)
                # Draw image
                length, info, img = detector.findDistance(center_point1, center_point2, img=img)

        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        exit()
cap.release()
cv2.destroyAllWindows()
