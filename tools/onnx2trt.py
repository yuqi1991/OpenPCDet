import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
a=(int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
device='cuda:0'

def GiB(val):
    return val * 1 << 30

def build_engine(onnx_path, using_half,engine_file,dynamic_input=None):
    trt.init_libnvinfer_plugins(None, '')
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1 # always 1 for explicit batch
        config = builder.create_builder_config()
        config.max_workspace_size = GiB(4)
        if using_half:
            config.set_flag(trt.BuilderFlag.FP16)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None

        if type(dynamic_input) == dict:
            profile = builder.create_optimization_profile();
            for k,v in dynamic_input.item():
                profile.set_shape(k,v[0],v[1],v[2])
            config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config)
        with open(engine_file, "wb") as f:
            f.write(engine.serialize())
        print('save trt engine success!!')

if __name__ == '__main__':
    onnx_path='all_sim.onnx'
    usinghalf=True
    batch_size=1
    engine_file='pointpillar.engine'

    #input name: min_shape, opt_shape, max_shape
    pfe_input_dict = {
        "voxel_features": [(1, 64, 4), (12000, 64, 4), (40000, 64, 4)],
        "voxel_num_points": [(1,), (12000,), (40000,)],
        "coords": [(1, 4), (12000, 4), (40000, 4)],
    }
    trt_engine=build_engine(onnx_path,usinghalf,engine_file,dynamic_input=pfe_input_dict)

    onnx_path='pp_anchor_trim.onnx'
    engine_file='pp_anchor.engine'
    trt_engine=build_engine(onnx_path,usinghalf,engine_file)



