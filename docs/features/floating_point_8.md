# Floating Point 8-bit

Floating Point 8-bit (FP8) enables faster and more memory-efficient inference by representing model weights and activations in 8-bit floating-point precision. The FP8 workflow involves three main stages:

- **Calibration**: Analyzing model weights and activations to determine optimal scaling factors and value ranges for accurate conversion.

- **Quantization**: Converting the model from higher precision, such as FP16, to FP8 using the calibrated ranges to minimize accuracy loss.

- **Inference**: Running the quantized model using FP8 computations, achieving faster execution with lower memory overhead while maintaining model quality.

## Calibration

Before running inference with FP8 precision on the Intel® Gaudi® HPU, the model must first be calibrated. Calibration generates the measurements, quantization files, and configuration data required for accurate FP8 inference. The vLLM Hardware Plugin for Intel® Gaudi® uses the [Intel® Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#inference-using-fp8) package to perform this calibration and enable efficient FP8 inference on the HPU. For more information about the calibration process and detailed setup instructions, refer to the [Calibration](../configuration/calibration/calibration.md) configuration guide.

## Quantization

Quantization trades off model precision for smaller memory footprint, allowing large models to be run on a wider range of devices. The Intel® Gaudi® Backend supports following quantization backends:

- Intel® Neural Compressor
- Auto_Awq
- Gptqmodel

For more information and detailed configuration recommendations for each backend, see the [Quantization and Inference](../configuration/quantization/quantization.md) configuration guide.

## Inference

The inference stage involves executing a trained model to generate predictions or outputs from new input data. After calibration and quantization, vLLM Hardware Plugin for Intel® Gaudi® runs the optimized model on supported hardware to deliver fast and accurate inference results. For more information and examples for different quantization backends, see the [Quantization and Inference](../configuration/quantization/quantization.md) guide.

### Qwen gated attention on Gaudi2

Qwen3Next and Qwen3.5 gated full-attention output projections use ordinary
rowwise dynamic FP8 scaling on Gaudi2, even when CGUID dynamic scaling is
enabled globally. This avoids a numerical anomaly in compiled graphs combining
the gate producer, CGUID scaling and FP8 GEMM. The projection-level policy also
applies to partial projections in the prefill boundary pipeline; other
projections and quantization methods keep their existing selection policy.

This workaround does not make FP8 inference equivalent to a higher-precision
model. Validate task quality and repeated execution for the model and workload
being deployed; moving graph breaks arbitrarily is not an equivalent workaround.
