import logging

logging.basicConfig(level=logging.DEBUG)

from copy import deepcopy

import cv2
import numpy as np
from rknnlite.api.rknn_lite import RKNNLite
import onnxruntime 
import time

class SegmentAnythingONNXRKNN:
    """Segmentation model using SegmentAnything"""

    def __init__(self, encoder_model_path, decoder_model_path) -> None:
        self.target_size = 1024
        self.input_size = (1024, 1024)

        self.encoder_session = RKNNLite()
        self.encoder_session.load_rknn(encoder_model_path)
        self.encoder_session.init_runtime()

        self.decoder_session = onnxruntime.InferenceSession(
            decoder_model_path, providers=["CPUExecutionProvider"]
        )

    def get_input_points(self, prompt):
        """Get input points"""
        points = []
        labels = []
        for mark in prompt:
            if mark["type"] == "point":
                points.append(mark["data"])
                labels.append(mark["label"])
            elif mark["type"] == "rectangle":
                points.append([mark["data"][0], mark["data"][1]])  # top left
                points.append(
                    [mark["data"][2], mark["data"][3]]
                )  # bottom right
                labels.append(2)
                labels.append(3)
        points, labels = np.array(points), np.array(labels)
        return points, labels

    def run_encoder(self, encoder_inputs):
        """Run encoder"""
        # output = self.encoder_session.run(None, encoder_inputs)
        start_time = time.time()
        output = self.encoder_session.inference(inputs=[encoder_inputs])
        print(f"Encoder Inference Time (ms): {(time.time() - start_time) * 1000}")
        image_embedding = output[0]
        return image_embedding

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def apply_coords(self, coords: np.ndarray, original_size, target_length):
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def run_decoder(
        self, image_embedding, original_size, transform_matrix, prompt
    ):
        """Run decoder"""
        input_points, input_labels = self.get_input_points(prompt)

        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate(
            [input_points, np.array([[0.0, 0.0]])], axis=0
        )[None, :, :]
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[
            None, :
        ].astype(np.float32)
        onnx_coord = self.apply_coords(
            onnx_coord, self.input_size, self.target_size
        ).astype(np.float32)

        # Apply the transformation matrix to the coordinates.
        onnx_coord = np.concatenate(
            [
                onnx_coord,
                np.ones((1, onnx_coord.shape[1], 1), dtype=np.float32),
            ],
            axis=2,
        )
        onnx_coord = np.matmul(onnx_coord, transform_matrix.T)
        onnx_coord = onnx_coord[:, :, :2].astype(np.float32)

        # Create an empty mask input and an indicator for no mask.
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(self.input_size, dtype=np.float32),
        }
        start_time = time.time()
        masks, _, _ = self.decoder_session.run(None, decoder_inputs)
        # masks, _, _ = self.decoder_session.run(inputs=[
        #     image_embedding, onnx_coord, onnx_label, onnx_mask_input, onnx_has_mask_input, np.array(self.input_size, dtype=np.float32)
        # ])
        print(f"Decoder Inference Time (ms): {(time.time() - start_time) * 1000}")
        # Transform the masks back to the original image size.
        inv_transform_matrix = np.linalg.inv(transform_matrix)
        transformed_masks = self.transform_masks(
            masks, original_size, inv_transform_matrix
        )

        return transformed_masks

    def transform_masks(self, masks, original_size, transform_matrix):
        """Transform masks
        Transform the masks back to the original image size.
        """
        output_masks = []
        for batch in range(masks.shape[0]):
            batch_masks = []
            for mask_id in range(masks.shape[1]):
                mask = masks[batch, mask_id]
                mask = cv2.warpAffine(
                    mask,
                    transform_matrix[:2],
                    (original_size[1], original_size[0]),
                    flags=cv2.INTER_LINEAR,
                )
                batch_masks.append(mask)
            output_masks.append(batch_masks)
        return np.array(output_masks)

    def encode(self, cv_image):
        """
        Calculate embedding and metadata for a single image.
        """
        original_size = cv_image.shape[:2]

        # Calculate a transformation matrix to convert to self.input_size
        scale_x = self.input_size[1] / cv_image.shape[1]
        scale_y = self.input_size[0] / cv_image.shape[0]
        scale = min(scale_x, scale_y)
        transform_matrix = np.array(
            [
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1],
            ]
        )
        cv_image = cv2.warpAffine(
            cv_image,
            transform_matrix[:2],
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_LINEAR,
        )

        encoder_inputs = cv_image.astype(np.float32)
        print(encoder_inputs.shape)
        image_embedding = self.run_encoder(encoder_inputs)
        return {
            "image_embedding": image_embedding,
            "original_size": original_size,
            "transform_matrix": transform_matrix,
        }

    def predict_masks(self, embedding, prompt):
        """
        Predict masks for a single image.
        """
        masks = self.run_decoder(
            embedding["image_embedding"],
            embedding["original_size"],
            embedding["transform_matrix"],
            prompt,
        )

        return masks

if __name__ == "__main__":
    encoder_model_path = "sam_vit_b_01ec64.pth.encoder.patched.onnx.rknn"
    decoder_model_path = "sam_vit_b_01ec64.pth.decoder.onnx"
    segmenter = SegmentAnythingONNXRKNN(encoder_model_path, decoder_model_path)

    image = cv2.imread("input.jpg")
    embedding = segmenter.encode(image)
    prompt = [
        {"type": "point", "data": [540, 512], "label": 0},
    ]
    masks = segmenter.predict_masks(embedding, prompt)
    print(masks.shape)
    
    # Save the masks as a single image.
    mask = np.zeros((masks.shape[2], masks.shape[3], 3), dtype=np.uint8)
    for m in masks[0, :, :, :]:
        mask[m > 0.0] = [255, 0, 0]

    # Binding image and mask
    visualized = cv2.addWeighted(image, 0.5, mask, 0.5, 0)

    # Draw the prompt points and rectangles.
    for p in prompt:
        if p["type"] == "point":
            color = (
                (0, 255, 0) if p["label"] == 1 else (0, 0, 255)
            )  # green for positive, red for negative
            cv2.circle(visualized, (p["data"][0], p["data"][1]), 10, color, -1)
        elif p["type"] == "rectangle":
            cv2.rectangle(
                visualized,
                (p["data"][0], p["data"][1]),
                (p["data"][2], p["data"][3]),
                (0, 255, 0),
                2,
            )

    cv2.imwrite("output.jpg", visualized)

