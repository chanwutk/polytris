import cv2

image = cv2.imread('/data/chanwutk/projects/polyis/track-results-0/jnc00.mp4.0.0.jpg')

for i  in range(1, 20):
    # draw a vertical line
    image = cv2.line(image, (128 * i - 2, 0), (128 * i - 2, 2000), (255, 255, 255), 4)
    image = cv2.line(image, (0, 128 * i - 2), (2500, 128 * i - 2), (255, 255, 255), 4)

# save
cv2.imwrite('split.jpg', image)