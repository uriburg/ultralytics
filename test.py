import torch
from ultralytics import YOLO


def main():
    device = torch.device("cuda")
    exp = "exp17"
    model_name = f'./results/{exp}/weights/last.pt'
    model = YOLO(model_name)

    img_size = 128
    model.predict("1.png", save=True, imgsz=img_size, conf=0.1, classes=[6, 8, 14])
    # Train
    #model.train(data="voc.yaml", project="results", name="exp", optimizer='SGD', imgsz=img_size, epochs=50, batch=512,
    #            classes=[6,8,14])

    ## Export
    #model.export(format="onnx", project="results", name="exp", imgsz=[img_size, img_size])

if __name__ == "__main__":
    main()