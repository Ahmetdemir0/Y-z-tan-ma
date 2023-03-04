import cv2  # opencv kütüphanesi eklendi
import numpy as np  # numpy kütüphanesi eklendi

taniyici = cv2.face.LBPHFaceRecognizer_create()  # yüz tanıyıcı oluşturuldu
taniyici.read('deneme/deneme.yml')  # okunacak dosya belirlendi
# sınıflandırıcı xml dosyası belirlendi
yolsiniflandirici = "haarcascade_frontalface_default.xml"

yuzsiniflandirici = cv2.CascadeClassifier(yolsiniflandirici)
font = cv2.FONT_HERSHEY_SIMPLEX  # yazı tipi belirlendi
vid_cam = cv2.VideoCapture(0)  # kamera tanımlandı

while True:

    ret, kamera = vid_cam.read()  # kamera okutuldu
    gri = cv2.cvtColor(kamera, cv2.COLOR_BGR2GRAY)  # gri ton eklendi
    yuzler = yuzsiniflandirici.detectMultiScale(
        gri, 1.2, 5)  # sınırlandırmalar tanımlandı

    for (x, y, w, h) in yuzler:
        # cerceve kalınlığı ve rengi belirlendi
        cv2.rectangle(kamera, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 3)
        # yazılacak isim değişkeni tanımlandı
        Id, conf = taniyici.predict(gri[y:y+h, x:x+w])
        print(Id)

        if (Id == 1):
            Id = "Ali Imran"  # Eğer 1.yüz verisi ise Ali İmran yaz
        elif (Id == 2):
            Id = "Algilanamadi"  # Eğer 2.yüz verisi ise Yüz Taninamadi yaz
        else:
            Id = "Yüz Taninamadi !"

       # çerçeve ebatları tanımlandı
        cv2.rectangle(kamera, (x-22, y-90), (x+w+22, y-22), (0, 255, 0), -1)
        # yazılacak isim adının ölçekleri , rengi ve kalınlığı belirlendi
        cv2.putText(kamera, str(Id), (x, y-40), font, 2, (255, 255, 255), 3)

    cv2.imshow('kamera', kamera)  # kamera göster komutu eklendi

    # kamera kalitesi belirlenip çıkış tuşu atandı .
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
vid_cam.release()  # kamera durduruldu
cv2.destroyAllWindows()  # tüm pencereler kapatıldı
