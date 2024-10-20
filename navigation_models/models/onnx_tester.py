import onnx
import torch
import time
from MiniTransformer import *
import torch
import torch.nn as nn
# from vit import *
import pickle
from test_backbone import *
# from MiniTransformer_Solver import Solver
import onnxruntime
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def convert_torch2onnx(torch_model, torch_input):
    onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
    # onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)
    # onnx_input = onnx.numpy_helper.from_array(torch_input.numpy())
    ONNX_PATH = "./transformer.onnx"
    print(f"Saving converted ONNX model to {ONNX_PATH}")
    onnx_program.save(ONNX_PATH)
    return ONNX_PATH, torch_input.numpy()


def benchmark_onnx_model(onnx_program: str, onnx_input: np.ndarray):
    start = time.time()
    ort_session = onnxruntime.InferenceSession(onnx_program, providers=['CPUExecutionProvider'])
    # ort_session = onnxruntime.InferenceSession(onnx_program, None)
    finish = time.time()
    print(f"Time to load ONNX model from disk: {finish-start:.2f} sec")
    
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    # print(input_name, output_name)
    times, outputs = [], []
    for i in range(10):
        start = time.time()
        result = ort_session.run([output_name], {input_name: onnx_input})
        prediction = np.array(result).squeeze()
        finish = time.time()
        times.append(finish-start)
        outputs.append(prediction)
    print(f"ONNX time to infer:{np.mean(times):.2f}    out={np.mean(outputs):.3f}")


def benchmark_torch_model():
    print("Loading trained transformer model....")
    PATH = "/p/rosbot/rosbotxl/models-yili/transformer-model2-training-output/transformer-head-9_1-0_23-Z67MQZ/model-statedict-ViT_B_16-224x224-100epoch-11Ksamples-epoch99.pt"
    start = time.time()
    backbone = ViT_B_16()
    head = LinearHead(in_features=768, out_features=1)
    backbone.backbone.heads = head
    state_dict = torch.load(PATH, map_location=torch.device('cpu'))
    backbone.load_state_dict(state_dict)
    backbone.eval()
    finish = time.time()
    print(f"Time to load Torch model from disk: {finish-start:.2f}")
    times, outputs = [], []
    test_input = torch.rand((1, 3, 224, 224))
    for i in range(10):
        start = time.time()
        out = backbone(test_input).detach().numpy().item()
        finish = time.time()
        # print(f"{out.shape=}, {out.item()=:.3f} ")
        times.append(finish-start)
        outputs.append(out)
    print(f"TORCH time to infer:{np.mean(times):.2f}    out={np.mean(outputs):.3f}")
    return backbone, test_input

# All times are in seconds
if __name__ == "__main__":
    backbone, torch_input = benchmark_torch_model()
    onnx_program, onnx_input = convert_torch2onnx(backbone, torch_input)
    print(f"Sample input: {type(onnx_input)}")
    benchmark_onnx_model(onnx_program, onnx_input)