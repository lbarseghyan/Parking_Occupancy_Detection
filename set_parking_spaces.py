import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector
from matplotlib.collections import PatchCollection
 

class SelectFromCollection(object):
    def __init__(self, axes):
        self.canvas = axes.figure.canvas
        self.poly = PolygonSelector(axes, self.onselect)
        self.ind = []

    def onselect(self, verts):
        global points_current
        points_current = verts
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.canvas.draw_idle()


def press_key(event):

    if event.key == 'f': 
        if len(set(points_current)) == 4:       
            polygons.append(Polygon(points_current))
            points_total.append(points_current)
        
    if event.key == 'e':
        print("Coordinates of saved parking spaces: "+str(points_total))
        selection.disconnect()
        with open(output_path, 'wb') as f:
            np.save(f, points_total)
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help="Path of the video")
    parser.add_argument('--output_path', help="Path of the output file",
                        default="parking_spaces.npy")
    args = parser.parse_args()

    output_path = args.output_path
    if not output_path.endswith(".npy"):
        output_path += ".npy"
      
    video = cv2.VideoCapture(args.video_path)
    if not video.isOpened():
        print("Error opening video file")
    else:
        while video.isOpened():
            success, frame = video.read()
            if success:
                video.release()
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print("> Select parking spaces by enclosing them within polygons.\n",
          "> To save the current polygon after marking it press 'f' , and then press 'q' to begin marking a new one.\n",
          "> To save your selections and end the program press 'e'\n")
    
    points_current = []
    points_total = []
    polygons = []
    
    while True:
        _, ax = plt.subplots()
        ax.imshow(frame_RGB)
    
        parking_spaces = PatchCollection(polygons, alpha=0.4)
        parking_spaces.set_facecolor('#0147AB')           
        ax.add_collection(parking_spaces)
            
        selection = SelectFromCollection(ax)
        plt.connect('key_press_event', press_key)
        plt.show()
        selection.disconnect()
