from ultralytics import YOLO

model= YOLO(r"C:\Users\USER\Desktop\Football_Anal\models\best.pt")


results = model.predict(r'C:\Users\USER\Desktop\Football_Anal\input_videos\new.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)