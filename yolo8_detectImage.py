import ultralytics
from ultralytics import YOLO
 #import opencv-python
import cv2
#changer des shapes des tableau : tableau -> vecteurs , matrice
import numpy as np


image = cv2.imread('img.png')
#importer model  contient 80 objects : person , bycl, fire...
model = YOLO('yolov8m.pt')
#retourne les noms des clesses dans le model
classes= model.names
print(classes)
#donner une image au model pour la predecte parmis ces classes 
results=model(image)
print(results)
#chaque resultat de results le faire un plot 'affichage des carreaux '
for i in results:
    im_array = i.plot()
#array=matrice
# uinit8  au 8 bit rendre pixel dans [0..255]
image_results = np.array(im_array,dtype=np.uint8)
#afficher image  
cv2.imshow('affichage image',image)
#detecte les objet de notre images : person , clock
cv2.imshow('affichage image_results',image_results)
cv2.waitKey(0)


