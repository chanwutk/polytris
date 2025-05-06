import cv2


cap = cv2.VideoCapture('./video-masked/jnc00.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # draw a 128x128 grid to frame (horizontal and vertical lines)
    for i in range(0, frame.shape[0], 128):
        cv2.line(frame, (0, i), (frame.shape[1], i), (255, 0, 0), 1)
    for i in range(0, frame.shape[1], 128):
        cv2.line(frame, (i, 0), (i, frame.shape[0]), (255, 0, 0), 1)

    # draw each grid index in the middle of the grid
    for i in range(0, frame.shape[0], 128):
        for j in range(0, frame.shape[1], 128):
            cv2.putText(frame, f'{i//128},{j//128}', (j+64, i+64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite('frame.jpg', frame)
    break