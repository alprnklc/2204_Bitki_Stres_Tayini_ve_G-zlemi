import cv2
import numpy as np

# Görüntüyü yükle
image = cv2.imread('lettuce9.jpg')

# ROI (Region of Interest) seçimi
print("Lütfen marul bölgesini seçmek için bir dikdörtgen çizin.")
roi = cv2.selectROI("Görüntü", image, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

# ROI koordinatları
x, y, w, h = map(int, roi)

# Maske oluşturma
mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Siyah bir maske
mask[y:y+h, x:x+w] = 255  # Seçilen alanı beyaz yap

# Maskeyi göster
cv2.imshow('Mask', mask)
cv2.imwrite('ground_truth.jpg', mask)  # Maskeyi kaydet
cv2.waitKey(0)
cv2.destroyAllWindows()
