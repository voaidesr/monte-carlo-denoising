# test if deps are installed correctly

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.circle(img, (100, 100), 50, (0, 255, 0), -1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Poetry + OpenCV + NumPy + Matplotlib")
plt.axis("off")
plt.show()