import onnxruntime as ort
import torch
import numpy as np
import sys
import os

path1 = "./inputs"
path2 = "./outputs"

os.makedirs(path1, exist_ok=True)
os.makedirs(path2, exist_ok=True)

session = ort.InferenceSession("fused.onnx")
n = 1
if len(sys.argv) > 1:
    n = int(sys.argv[1])

for i in range(n):
    data = torch.rand(1, 32, 160, 160).numpy()

    #session = ort.InferenceSession("C3.onnx")
    outputs = session.run(None, {"input_C3": data})

    np.savetxt(f"inputs/input_fused_{i}.txt", np.reshape(data, (32, 25600)), \
            fmt='%.7f')
    np.savetxt(f"outputs/output0_fused_{i}.txt", np.reshape(outputs[0], (16, 25600)), \
            fmt='%.7f')
    np.savetxt(f"outputs/output1_fused_{i}.txt", np.reshape(outputs[1], (16, 25600)), \
            fmt='%.7f')
    #print("[", len(output), ", ", len(output[0]), ", ", len(output[0][0]), \
    #        ", ", len(output[0][0][0]), ", ", len(output[0][0][0][0]), "]")

    #print(output[0][0][0][0][0])
