import onnx

model = onnx.load("live_model(html+js)/CMM-Yolo11.onnx")
print("Inputs:")
for inp in model.graph.input:
    print(inp.name, inp.type.tensor_type.shape)

print("\nOutputs:")
for out in model.graph.output:
    print(out.name, out.type.tensor_type.shape)
