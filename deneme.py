import cv2  #opencv kütüphanesini atadık
import numpy as np  #numpy kütüphanesini atadık kullanılmıyor ama geliştirirken lazım olabilir
import sys
import pickle #pickle ile ufak tefek veritabanı tarzı kayıtlarımızı yapacağız


yuzde=100.0
cascPath = 'siniflandiricilar/haarcascade_frontalface_alt2.xml'   #yüzün tanınmasını sağlayan matematiksel datasheetler
faceCascade = cv2.CascadeClassifier(cascPath)            #opencv ile cascade imizi bağladık
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("egitici.yml")
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:                 #pickle dan yani minik veritabanımızdan kaç tane yüzün kayıtlı olduğunu ve isimleri çekiyoruz
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}
video_capture = cv2.VideoCapture(0)                   #opencv kütüphanesi yardımı ile kameramızı açıyoruz

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()               #ret ve frame değişkenlerine kameradan alınan her kareyi alıyoruz

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #renk demek 0-255 arası demek biz daha basitleştirmek adına sadece gri tonlarına dönüştürüyoruz görüntüyü
                                                   #fakat görüntümüz bize renkli görünecek sadece arkada anlık işlem yapılan görüntü gri tonlarına dönüşecek
    faces = faceCascade.detectMultiScale(          #belirlenecek yüzlerin özelliklerini belirtiyoruz
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:                   #görüntüdeki yüzlerin yüksekliklerini ve genişliklerini alıyoruz
        baba_gray = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)    #genişlik ve yükseklik dahilinde yüzleri kırmızı bir kare içine alıyoruz

    # Display the resulting frame
        id_, conf = recognizer.predict(baba_gray)        #tanınması gereken yüzlerin confidence yani güven faktörünü ve hangi id ile tanındığını tanımlıyoruz
        if conf < 60:                                    #güven olayı biraz ters işliyor o yüzden karmaşıklık %60 in altındaysa şartları sağla dedik
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            name1=""
            color = (0,0,255)
            stroke = 2
            cv2.putText(frame, name +" %"+ str(yuzde-conf)[:2], (x,y+h+30), font, 1, color, stroke, cv2.LINE_AA)           #eğer yüz tanındıysa kırmızı karenin altında kime ait olduğunu bize söyle dedik
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name=""
            name1 = "Bilinmiyor"
            color = (0, 0, 255)
            stroke = 2
            cv2.putText(frame, name1, (x, y + h + 30), font, 1, color, stroke, cv2.LINE_AA)     #eğer tanınmadıysa bilinmiyor yaz
            cv2.putText(frame, name, (x, y + h + 30), font, 1, color, stroke, cv2.LINE_AA)      #ayrıca daha önce tanınan yüzün etiketini temizle

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):                                          # "Q" tuşuna basılana kadar programı kapatma
        break

# Eğer "Q" tuşuna basılırsa aşağıdaki kodlar çalışacak ve programa ait her şeyi sonlandıracak
video_capture.release()
cv2.destroyAllWindows()
