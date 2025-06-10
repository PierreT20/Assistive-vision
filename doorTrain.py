import torch
import torchvision.models as models
import torch.nn as nn
import coremltools as ct

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 3)

state_dict = torch.load("door_classifier.pt", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)

model.eval()

dummy_input = torch.randn(1, 3, 128, 128)

torch.onnx.export(
    model,
    dummy_input,
    "door_classifier.onnx",
    input_names=["image"],
    output_names=["output"],
    dynamic_axes={"image": {0: "batch_size"}}
)

coreml_model = ct.converters.onnx.convert(
    model="door_classifier.onnx",
    minimum_ios_deployment_target="13",
    image_input_names=["image"],
    image_scale=1.0 / 255.0,
    red_bias=-1,
    green_bias=-1,
    blue_bias=-1
)

coreml_model.save("door_classifier.mlmodel")

print("Model trained successfully!")