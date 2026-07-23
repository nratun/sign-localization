from ultralytics import YOLO
import sys

'''
Trains a model using customized parameters for most optimal output.
A pretrained YOLO Object Bounding Boxes model is used.
'''
def train():
    # Load pretrained OBB model
    model = YOLO("models/yolo26s-obb.pt")

    # Train model with custom parameters
    model.train(
        data="dataset/data.yaml",

        # Configuration
        epochs=150,
        imgsz=1024,
        batch=-1,
        device=0,
        cache=True,

        # Save
        project="runs/train",
        name="r-small",

        # Augmentation
        degrees=15,
        translate=0.15,
        multi_scale=0.25,
        shear=5,
        perspective=0.0006,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=15,

        # Color augmentation
        hsv_h=0.02,
        hsv_s=0.75,
        hsv_v=0.6,

        # Convergence
        patience=20,
        cos_lr=True,
    )

def validate():
    # Load trained model
    model = YOLO("runs/obb/runs/train/r/weights/best.pt")

    # Validate the model
    metrics = model.val()
    print("Map50-95:\t", metrics.box.map)    # map50-95
    print("Map50:\t", metrics.box.map50)       # map50
    print("Map75:\t", metrics.box.map75)       # map75

if __name__ == "__main__":
    command = sys.argv[1].lower()
    if command == "train":
        train()
    elif command == "validate":
        validate()