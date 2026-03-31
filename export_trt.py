import tensorrt as trt
import os

def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
    
    if builder.platform_has_tf32:
        config.set_flag(trt.BuilderFlag.TF32)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    # Optimization Profiles for dynamic input size
    profile = builder.create_optimization_profile()
    profile.set_shape("image", (1, 3, 112, 112), (1, 3, 322, 322), (1, 3, 1022, 1022))
    config.add_optimization_profile(profile)

    print(f"Building engine: {engine_file_path}...")
    engine_bytes = builder.build_serialized_network(network, config)
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
    print("Done!")

if __name__ == "__main__":
    build_engine("anycalib_nn.onnx", "anycalib_nn.engine")
