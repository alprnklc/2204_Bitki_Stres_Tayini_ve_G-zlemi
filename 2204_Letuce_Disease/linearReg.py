from sklearn.linear_model import LinearRegression
import numpy as np

# Beyaz piksel oranı ve sınıflar (0: düşük, 1: yüksek)
X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])  # Özellik: beyaz piksel oranı
y = np.array([0, 0, 1, 1, 1])  # Hedef: sınıflar

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X, y)

# Yeni veri üzerinde tahmin yap
test_data = np.array([[0.4], [0.6], [0.8]])  # Beyaz piksel oranları
predictions = model.predict(test_data)

# Tahmin sonuçlarını sınıflara dönüştürmek için eşik uygula (örneğin, 0.5)
threshold = 0.5
classified_predictions = (predictions >= threshold).astype(int)

# Sonuçları yazdır
for i, (input_value, prediction, classified) in enumerate(zip(test_data, predictions, classified_predictions)):
    print(f"Input: {input_value[0]:.2f}, Prediction: {prediction:.2f}, Classified: {classified}")