import cv2

# Kamerayı başlat
cap = cv2.VideoCapture(1)  # 0 varsayılan kamerayı temsil eder

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# Anlık görüntü alma
while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamadı!")
        break

    # Görüntüyü ekranda göster
    cv2.imshow('Live Camera', frame)

    # 's' tuşuna basıldığında anlık görüntüyü kaydet
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('snapshot.jpg', frame)  # Görüntüyü kaydet
        print("Anlık görüntü kaydedildi!")
        break

    # 'q' tuşuna basıldığında çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Çıkış yapılıyor.")
        break

# Kamerayı ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
