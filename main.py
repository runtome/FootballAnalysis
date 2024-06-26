import numpy as np 
import cv2 as cv

from trackers import Tracker

def main():
  #Path to video
  video_path = "input/55c9d1_4.mp4"

  #inttial Tracker
  tracker = Tracker('models/best.pt')

  cap = cv.VideoCapture(video_path)
  if not cap.isOpened():
    print("Cannot open camera")
    exit()

  while True:
    #Cap Frame by Frame 
    ret , frame = cap.read()

    #  if frame is read correctly ret is True
    if not ret :
      print("Cant recieve frame (Stream end ? )")
      break
    
    frame = tracker.draw_player(frame)


    # Display the resulting frame
    cv.imshow('frame', frame)
    
    if cv.waitKey(24) == ord('q'):
        break
  
  # When everything done, release the capture
  cap.release()
  cv.destroyAllWindows()

if __name__ == "__main__":
  main()
