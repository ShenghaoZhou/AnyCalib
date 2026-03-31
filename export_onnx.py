import argparse
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
        # Note: 'fov_field' is just an alias for 'tangent_coords'
        rays = out["rays"].permute(0, 2, 3, 1).flatten(1, 2)
        tangent_coords = out["tangent_coords"].permute(0, 2, 3, 1).flatten(1, 2)
        return rays, tangent_coords

def export_to_onnx(model_id="anycalib_pinhole", output_path="anycalib_nn.onnx"):
    print(f"Exporting model {model_id} to {output_path}...")
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
                      'rays': {0: 'batch_size', 1: 'n_rays'},
                      'tangent_coords': {0: 'batch_size', 1: 'n_rays'}}
    )
    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="anycalib_pinhole", 
                        choices=["anycalib_pinhole", "anycalib_dist", "anycalib_gen", "anycalib_edit"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"{args.model_id}.onnx"
        
    export_to_onnx(args.model_id, args.output)
