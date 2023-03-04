import cv2, os #opencv ve os kütüphanesi eklendi
import numpy as np #numpy kütüphanesi eklendi
from PIL import Image #image kütüphanesi eklendi

taniyici = cv2.face.LBPHFaceRecognizer_create() # yüz tanıyıcı oluşturuldu
# sınıflandırıcı xml dosyası eklendi
dedektor = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLabels(path): #görüntüler ve tanımlar için yol atandı
    #resim yolunu bulmak için döngü oluşturuldu
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    yuzornekleri = [] #değişken tanımlandı
    isimler = [] #değişken tanımlandı
    for imagePaths in imagePaths: # for döngüsü oluşturuldu
        #resmin okunacağı dosyayı açmak için PIL_img 'ten yararlanıldı
        PIL_img = Image.open(imagePaths).convert('L')
        img_numpy = np.array(PIL_img,'uint8') # pozitif tamsayı olduğu belirtildi
        id = int(os.path.split(imagePaths)[-1].split(".")[1]) #ismin yazılacağı yer belirlendi
        print(id)
        # yüz ölçeklerinin belirleneceği kod satırı atandı
        yuzler = dedektor.detectMultiScale(img_numpy)

        for (x,y,w,h) in yuzler: #yüz çerçevesi için değişkenler for döngüsüne eklendi 

            yuzornekleri.append(img_numpy[y:y+h,x:x+w]) #karakter dizisi belirlendi
            isimler.append(id) # karakter dizisi okutuldu

    return yuzornekleri,isimler

yuzler,isimler = getImagesAndLabels('veri') # verilerin bulunduğu klasör adı girildi
taniyici.train(yuzler, np.array(isimler)) #tanıyıcı deneme oluşturuldu
taniyici.save('deneme/deneme.yml') #kaydedilecek klasör belirlendi
