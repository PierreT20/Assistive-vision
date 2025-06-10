import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.serialization import add_safe_globals, safe_globals
from ultralytics.nn.tasks import DetectionModel


def build_door_classifier(num_classes=3):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class CombinedModel(nn.Module):
    def __init__(self, door_model_path, yolo_model_path):
        super(CombinedModel, self).__init__()

        self.door_model = build_door_classifier()
        self.door_model.load_state_dict(
            torch.load(door_model_path, map_location=torch.device('cpu'))
        )
        self.door_model.eval()

        add_safe_globals([DetectionModel])
        with safe_globals([DetectionModel]):
            checkpoint = torch.load(
                yolo_model_path, map_location=torch.device('cpu'), weights_only=False
            )
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.yolo_model = checkpoint["model"]
        else:
            self.yolo_model = checkpoint
        self.yolo_model.eval()

    def forward(self, x):
        yolo_output = self.yolo_model(x)
        door_output = self.door_model(x)
        return {"yolo": yolo_output, "door": door_output}


if __name__ == '__main__':
    combined_model = CombinedModel("door_classifier.pt", "yolo11n.pt")
    torch.save(combined_model, "combined_model.pt")