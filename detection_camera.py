import cv2
from ultralytics import YOLO
import numpy as np
#utilisation de camera 0 qui est integre , si autre 1 ou 3eme camera 3
model = YOLO('yolov8n.pt')
camera = cv2.VideoCapture(0)

#si camera active
#ret : true or false : camera active or not
#image qui capte l 'image de camera par 1 s : capture sucessive
while True:
    ret,image = camera.read()
    results=model(image)
    for r in results:
        #stock l'image dans im_array
        im_array = r.plot()
    image_result = np.array(im_array,dtype=np.uint8)
    cv2.imshow("affichage detecte image camera",image_result)

    cv2.imshow('affichage camera',image)
    cv2.waitKey(1)

#ctrl c pour quitter camera 
#appliquer le detection sur l'image capturé apres on l'affiche le resultat 