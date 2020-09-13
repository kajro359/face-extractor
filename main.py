from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2 as cv
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import glob
from img_from_roi import save_roi
import help_functions as hf
from sort import *
from face_detection import select


def parseArgs():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, help="path to input video file")
    ap.add_argument("-d", "--display", action="store_true", default=False,
            help="display frames")
    ap.add_argument("-m", "--method", required=False, default="ssd",
            help="method to use for detection")
    ap.add_argument("-o", "--output", type=str, help="path to output folder", default="output")
    return vars(ap.parse_args())


def getVidName(path):
    ind0 = path[::-1].find("/")
    ind1 = path[::-1].find(".") - 1
    return(path[-ind0:-ind1])


def dict2csv(dict, fName):
    with open(fName + '.csv', 'w') as f:
        for key in dict:
            f.write('{},{}\n'.format(key, dict[key]))


def main():

    args = parseArgs()
    if(not os.path.exists(args["output"])): os.mkdir(args["output"])
    detector, detect_faces = select(args['method'])

    # Open videostream
    webcam = False
    if not args.get("video", False): # grab from webcam if no file supplied
        print("starting video stream")
        webcam = True
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        vidName = 'webcam'
    else:
        vidName = getVidName(args['video'])
        vs = cv.VideoCapture(args["video"])
    
    mot_tracker = Sort()
    img_counter = {}
    dets = {}

    # time measure
    n = 0 # frame counter
    t0 = time.time()
    while True:
        if not args.get("video", False): #webcam used
            frame = vs.read()
            hasFrame = True
        else:
            hasFrame, frame = vs.read()            
        
        
        if hasFrame:
            faces = detect_faces(frame, detector)
            # update dets dict
            dets[n] = len(faces) 
            n = n + 1

            frame_cpy = frame

            # Update tracked bounding boxes
            track_bbs_ids = mot_tracker.update(faces)
            # Remove negative coordinates
            track_bbs_ids[track_bbs_ids < 0] = 0

            # Visualize and save face images
            for (x1, y1, x2, y2, id_no) in track_bbs_ids.astype('int'):
                if(id_no not in img_counter): img_counter[id_no] = 0
                else: img_counter[id_no] = img_counter[id_no] + 1
             
                save_roi(frame, np.array([[x1, y1, x2-x1, y2-y1]]),
                        '_' + str(id_no).zfill(4) + '_' + 
                        str(img_counter[id_no]).zfill(3),
                        args["output"])
                print("Saved img face_" + str(id_no).zfill(3) + '_' + 
                        str(img_counter[id_no]).zfill(3))
                
                if(args["display"]):
                    cv.rectangle(frame_cpy, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    if(y1 < 20): (t1,t2) = (x1,20)
                    else: (t1,t2) = (x1,y1)
                    cv.putText(frame_cpy, str(id_no).zfill(3), 
                            (t1,t2), 
                            cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                # save frame with bb's in it
                if not os.path.exists(os.path.join(args["output"], "stills")) : 
                    os.mkdir(os.path.join(args["output"], "stills"))
                cv.imwrite(os.path.join(args["output"], "stills", "detsframe{}.jpg".format(n)), 
                        frame_cpy)
                # print(os.path.join(args["output"], "stills", "detsframe{}.jpg".format(n)))
            
            if(args["display"]): cv.imshow("Frame", frame_cpy)

            
        key = cv.waitKey(1) & 0xFF

        if key == ord("q") or not hasFrame:
            cv.destroyAllWindows()
            if(webcam): 
                vs.stop()
            else:
                vs.release()
            break
    
    t1 = time.time()
    dict2csv(dets, 'dets-' + vidName)
    print('Detections in each frame: ')
    print(dets)
    print(img_counter)
    print("Processed {} frames in {} seconds".format(n, t1-t0))
    print("Frames per second: {}".format(n/(t1-t0)))
    # for ID in img_counter:
    #     if img_counter[ID] < 10:
    #         #remove faces that have fewer than 10 imgs.
    #         for file in glob.glob(os.path.join(args["output"], 
    #             'face_' + str(ID).zfill(3) + '*')):
    #             os.remove(file)

    

if __name__ == "__main__":
    main()
