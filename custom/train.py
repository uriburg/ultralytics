import torch
from ultralytics import YOLO


def main():
    device = torch.device("cuda")
    load = False
    exp = "exp21"
    if load:
        model_name = f'./results/{exp}/weights/last.pt'
        model = YOLO(model_name)
    else:
        model_name = f'./custom/model.yaml'
        model = YOLO(model_name)

    img_size = 128

    # Train
    model.train(data="custom/voc.yaml", project="results", name="exp", optimizer='SGD', imgsz=img_size, epochs=500, device=0, batch=-1,
                classes=[6,8,14])#, workers=12)

    ## Export
    #model.export(format="onnx", project="results", name="exp", imgsz=[img_size, img_size])

if __name__ == "__main__":
    main()