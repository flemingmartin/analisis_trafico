import pafy
import cv2

url = "https://www.youtube.com/watch?v=1EiC9bvVGnk&ab_channel=SeeJacksonHole"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

capture = cv2.VideoCapture(best.url)
while True:
    grabbed, frame = capture.read()
    cv2.imshow("opaa",frame)

    cv2.waitKey(10)