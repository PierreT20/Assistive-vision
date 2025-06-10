import torch
import coremltools as ct
from torch import nn
import torchvision.models as models

try:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    state_dict = torch.load("door_classifier.pt")
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
        inputs=[ct.ImageType(name="image", shape=(1, 3, 128, 128))],
        image_scale=1.0 / 255.0,
        red_bias=-1,
        green_bias=-1,
        blue_bias=-1
    )

    coreml_model.save("door_classifier.mlmodel")

    print("Model conversion completed successfully!")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("\nPlease make sure you have the following packages installed with the correct versions:")
    print("pip install numpy==1.24.3 torch torchvision coremltools")