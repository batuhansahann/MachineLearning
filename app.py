from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

# Flask uygulaması oluşturma
app = Flask(__name__)

# Model dosyasının konumu
model_dosyasi = os.path.join(os.path.dirname(__file__), "best_random_forest_model.pkl")

# Modeli yükle
try:
    with open(model_dosyasi, "rb") as dosya:
        model = pickle.load(dosya)
    print("Model başarıyla yüklendi!")
except FileNotFoundError:
    print(f"Model dosyası bulunamadı: '{model_dosyasi}'")

# CSV dosyasının konumu (Aynı klasör içinde olduğunu varsayıyoruz)
csv_dosya = os.path.join(os.path.dirname(__file__), "usa_visa.csv")

# CSV dosyasını yükle
try:
    veriler = pd.read_csv(csv_dosya)
    print("CSV dosyası başarıyla yüklendi!")
except FileNotFoundError:
    print(f"CSV dosyası yüklenirken hata oluştu: '{csv_dosya}' bulunamadı.")

# Ana sayfa route'u
@app.route("/", methods=["GET"])
def home():
    return "Model Başarıyla Yüklendi ve Çalışıyor!"

# API Route'u - Tahmin yapmak için
@app.route("/tahmin", methods=["POST"])
def tahmin():
    try:
        # İstekten JSON verisini al
        girdi_verileri = request.json

        # Verileri DataFrame'e dönüştür
        girdi_df = pd.DataFrame([girdi_verileri])

        # Model için uygun özellikleri seç
        X = girdi_df[['Saat_Sadece', 'Gün', 'Ay', 'Yıl', 'Dakika', 'Weekday', 'Month', 'Day']]

        # Tahmin yap
        tahmin_sonucu = model.predict(X)[0]

        # Yanıtı JSON formatında döndür
        return jsonify({"Tahmin Edilen Hour_Block": int(tahmin_sonucu)})

    except Exception as e:
        return jsonify({"hata": str(e)})

# Flask uygulamasını çalıştır
if __name__ == "__main__":
    app.run(debug=True)
