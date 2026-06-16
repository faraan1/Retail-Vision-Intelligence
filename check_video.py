import cv2
cap = cv2.VideoCapture('test_video.mp4')
ret, frame = cap.read()
if ret:
    cv2.imwrite('video_frame.jpg', frame)
    print(f"Frame saved! Size: {frame.shape}")
cap.release()