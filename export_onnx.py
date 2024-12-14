from PIL import Image
import torch
from gazelle.model import get_gazelle_model
import onnx
from onnxsim import simplify

"""
"gazelle_dinov2_vitb14": gazelle_dinov2_vitb14, gazelle_dinov2_vitb14.pt
"gazelle_dinov2_vitl14": gazelle_dinov2_vitl14, gazelle_dinov2_vitl14.pt
"gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout, gazelle_dinov2_vitb14_inout.pt
"gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout, gazelle_dinov2_vitl14_inout.pt

    def get_transform(self, in_size):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            ),
            transforms.Resize(in_size),
        ])
"""

models = {
    "gazelle_dinov2_vitb14": ["gazelle_dinov2_vitb14.pt", False],
    # "gazelle_dinov2_vitl14": ["gazelle_dinov2_vitl14.pt", False],
    # "gazelle_dinov2_vitb14_inout": ["gazelle_dinov2_vitb14_inout.pt", True],
    # "gazelle_dinov2_vitl14_inout": ["gazelle_dinov2_vitl14_inout.pt", True],
}

for m, params in models.items():
    model, transform = get_gazelle_model(model_name=m, onnx_export=True)
    model.load_gazelle_state_dict(torch.load(params[0], weights_only=True))
    model.eval()
    model.cpu()

    # image = Image.open("path/to/image.png").convert("RGB")
    # input = {
    #     # tensor of shape [1, 3, 448, 448]
    #     "images": transform(image).unsqueeze(dim=0).cpu(),
    #     # list of lists of bbox tuples
    #     # [(xmin, ymin, xmax, ymax)], 0.0-1.0 norm
    #     "bboxes": [[(0.1, 0.2, 0.5, 0.7)]]
    # }

    # with torch.no_grad():
    #     output = model(input)
    # # access prediction for first person in first image. Tensor of size [64, 64]
    # predicted_heatmap = output["heatmap"][0][0]
    # # in/out of frame score (1 = in frame) (output["inout"] will be None  for non-inout models)
    # predicted_inout = output["inout"][0][0]


    onnx_file = f"{params[0]}_1x3x448x448.onnx"
    images = torch.randn(1, 3, 448, 448).cpu()
    bboxes = torch.randn(1, 1, 4).cpu()
    if not params[1]:
        outputs = [
            'heatmap',
        ]
    else:
        outputs = [
            'heatmap',
            'inout',
        ]

    torch.onnx.export(
        model,
        args=(images, bboxes),
        f=onnx_file,
        opset_version=14,
        input_names=[
            'images',
            'bboxes_x1y1x2y2',
        ],
        output_names=outputs,
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
