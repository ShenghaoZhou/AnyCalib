import torch
import torch.nn as nn
from anycalib.model.anycalib_pretrained import AnyCalib

class AnyCalibNN(nn.Module):
    def __init__(self, model_id="anycalib_pinhole"):
        super().__init__()
        # Initialize full model to load weights
        full_model = AnyCalib(model_id=model_id)
        self.backbone = full_model.backbone
        self.decoder = full_model.decoder
        self.head = full_model.head
        self.backbone.model.interpolate_offset = 0.0 # Fix for dynamic shape export
        self.eval()

    def forward(self, image):
        # Neural network part of the forward pass
        out = self.backbone(image)
        out = self.decoder(out)
        out = self.head(out)
        return out["rays"], out["tangent_coords"]

def export_to_onnx(model_id="anycalib_pinhole", output_path="anycalib_nn.onnx"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnyCalibNN(model_id).to(device)
    
    # Example input size (must be divisible by 14 due to DINOv2)
    # Common training resolution is 102400 pixels (~320x320)
    dummy_input = torch.randn(1, 3, 322, 322).to(device)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=18, # Use 18 as recommended by the exporter
        do_constant_folding=True,
        input_names=['image'],
        output_names=['rays', 'tangent_coords'],
        dynamic_axes={'image': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'rays': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'tangent_coords': {0: 'batch_size', 2: 'height', 3: 'width'}}
    )
    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    export_to_onnx()
