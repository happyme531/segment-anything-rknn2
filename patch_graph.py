import onnx_graphsurgeon as gs
import onnx
import numpy as np

# Load the ONNX model
graph = gs.import_onnx(onnx.load("check3_fuse_ops.onnx"))

count=0
# Iterate through all nodes in the graph
for node in graph.nodes:
    # Check if the node is a Reshape operator
    if node.op == 'Reshape':
        # Get the shape input of the Reshape node
        shape_input = node.inputs[1]
        
        # Check if the shape input is a constant (which it should be for static reshapes)
        if isinstance(shape_input, gs.Constant):
            current_shape = shape_input.values
            
            # Check if it's a 5D reshape with the target shape [12,64,64,...,...]
            if len(current_shape) == 5 and current_shape[0] == 12 and current_shape[1] == 64 and current_shape[2] == 64:
                # Modify the shape to [12,4096,...,...]
                new_shape = np.array([12, 4096, current_shape[3], current_shape[4]], dtype=np.int64)
                print(f"Patched {current_shape} -> {new_shape}")

                # Update the shape input with the new shape
                shape_input.values = new_shape
                count = count + 1
                # print(f"Patched {node}")


            # Check if it's a 5D reshape with the target shape [300,14,14,...,...]
            if len(current_shape) == 5 and current_shape[0] == 300 and current_shape[1] == 14 and current_shape[2] == 14:
                # Modify the shape to [300,196,...,...]
                new_shape = np.array([300, 196, current_shape[3], current_shape[4]], dtype=np.int64)
                print(f"Patched {current_shape} -> {new_shape}")
                
                # Update the shape input with the new shape
                shape_input.values = new_shape
                count = count + 1
                # print(f"Patched {node}")

graph.cleanup().toposort()
print(f"Patched {count} nodes.")

model = gs.export_onnx(graph)

# Delete old shape information from the model
for value_info in model.graph.value_info:
    value_info.type.tensor_type.ClearField('shape')
    
# Save the modified model
onnx.save(model, "sam_vit_b_01ec64.pth.encoder.patched.onnx")

print("Saved as 'sam_vit_b_01ec64.pth.encoder.patched.onnx'")