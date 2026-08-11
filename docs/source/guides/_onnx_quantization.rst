================================
ONNX Quantization - Linux (Beta)
================================

ModelOpt provides ONNX quantization that works together with `TensorRT Explicit Quantization (EQ) <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-quantization>`_. The key advantages offered by ModelOpt's ONNX quantization:

#. Easy to use for non-expert users.
#. White-box design allowing expert users to customize the quantization process.
#. Better support for vision transformers.

Currently ONNX quantization supports FP8, INT4 and INT8 quantization.

.. note::

    ModelOpt ONNX quantization generates new ONNX models with QDQ nodes following TensorRT rules.
    For real speedup, the generated ONNX should be compiled into TensorRT engine.

.. _ort_ep_requirements:

Requirements
============

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Execution Provider
     - Requirements
   * - CPU
     - * Default
   * - CUDA
     - * Add ``libcudnn*.so*`` path to ``LD_LIBRARY_PATH`` (check required cuDNN version `here <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements>`_)
   * - TensorRT
     - * Same requirement as CUDA EP
       * TensorRT >= 10.0 (add TensorRT ``lib/`` path to ``LD_LIBRARY_PATH`` and  install python wheel). Please refer to `TensorRT 10.0 download link <https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz>`_.

Apply Post Training Quantization (PTQ)
======================================

PTQ should be done with a calibration dataset. If calibration dataset is not provided, ModelOpt will use random scales for the QDQ nodes.

Prepare calibration dataset
---------------------------
ModelOpt supports npz/npy file as calibration data format and that numpy file should be a dictionary with keys as model input names and values as numpy arrays.

.. code-block:: python

    # Example numpy file for single-input ONNX
    calib_data = np.random.randn(batch_size, channels, h, w)
    np.save("calib_data.npy", calib_data)

    # Example numpy file for single/multi-input ONNX
    # Dict key should match the input names of ONNX
    calib_data = {
        "input_name": np.random.randn(*shape),
        "input_name2": np.random.randn(*shape2),
    }
    np.savez("calib_data.npz", calib_data)



Call PTQ function
-----------------
.. code-block:: python

    import modelopt.onnx.quantization as moq

    calibration_data = np.load(calibration_data_path)

    moq.quantize(
        onnx_path=onnx_path,
        calibration_data=calibration_data,
        output_path="quant.onnx",
        quantize_mode="int8",
    )

Optionally enable Autotune for more optimized Q/DQ placement. Note that this will likely increase the time required to calibrate the model.

.. code-block:: python

    moq.quantize(
        ...
        # Default Autotune settings, can be tuned with the autotune_* arguments below.
        autotune=True,
    )

Alternatively, you can call PTQ function in command line:

.. argparse::
   :module: modelopt.onnx.quantization.__main__
   :func: get_parser
   :prog: python -m modelopt.onnx.quantization


If the model contains custom ops, enable calibration with the TensorRT Execution Provider backend with CUDA and CPU fallback (``--calibration_eps trt cuda:0 cpu``) and, if relevant, provide the location of the TensorRT plugin in ``.so`` format via the ``--trt_plugins`` flag.

By default, after running the calibration, the quantization tool will insert the QDQ nodes by following TensorRT friendly QDQ insertion algorithm. Users can change the default quantization behavior by tweaking the API params like op_types_to_quantize, op_types_to_exclude etc. See the :meth:`modelopt.onnx.quantization.quantize() <modelopt.onnx.quantization.quantize>` for details.


Quantization Sensitivity Scan
=============================

For hybrid CNN + Transformer models it is often unclear which ops or nodes destroy accuracy when
quantized. The :func:`modelopt.onnx.quantization.sensitivity.score` primitive ranks quantizable
targets (op types or individual nodes) by the proxy metric between the FP16 reference and the per-
target quantized activations, so a downstream picker can decide which ops to keep at higher
precision. The primitive reuses :func:`modelopt.onnx.quantization.quantize` internally for each
per-target probe, so scales are properly calibrated (not autotune's placement-only descriptors).

Python API:

.. code-block:: python

    from modelopt.onnx.quantization.sensitivity import score

    result = score(
        onnx_path="coatnet-0.onnx",
        calibration_data="imagenet_calib_500.npz",
        granularity="op_type",  # or "node"
        metric="kl_div",
        target_precision="int8",
    )
    # result["scores"] is a dict {op_type_or_node_name: metric_value}, higher = more sensitive.

Command line:

.. code-block:: bash

    # Op-type ranking with real calibration data (headline mode; ~2-3 min on CoAtNet-0)
    python -m modelopt.onnx.quantization.sensitivity \
        --onnx_path coatnet-0.onnx \
        --calibration_data_path imagenet_calib_500.npz \
        --granularity op_type

    # Per-node ranking (deep dive; ~30-60 min on CoAtNet-0)
    python -m modelopt.onnx.quantization.sensitivity \
        --onnx_path coatnet-0.onnx \
        --calibration_data_path imagenet_calib_500.npz \
        --granularity node

Rendered ranking (CoAtNet-0, real calibration)::

    Sensitivity scan (int8 / kl_div / op_type):
      LayerNormalization    0.402  <-- highest impact
      Sigmoid               0.318
      Mul                   0.271
      MatMul                0.187
      Add                   0.043
      Conv                  0.012  <-- lowest impact
    Wrote coatnet-0.sensitivity.json

.. note::

    Omitting ``--calibration_data_path`` falls back to synthetic random inputs. Absolute scores are
    then directional-only and must not be paired with absolute thresholds -- attention-heavy models
    are the highest-risk degradation case because random Q times K^T produces near-uniform softmax
    that hides real-input MHA quantization pathology. The ``calibration_source`` field of the
    output JSON records which mode was used.

The per-node granularity depends on the ``--nodes_to_include <regex>`` filter added to the main
quantize CLI; the same flag is available as a symmetric mirror of ``--nodes_to_exclude`` for any
future single-node inspection workflow.

Deploy Quantized ONNX Model
===========================


``trtexec`` is a command-line tool provided by TensorRT. Typically, it's within the ``/usr/src/tensorrt/bin/`` directory. Below is a simple command to compile the quantized onnx model generated by the previous step into a TensorRT engine file.

.. code-block:: bash

    trtexec --onnx=quant.onnx --saveEngine=quant.engine --best

Compare the performance
=======================

The following command will build the engine using fp16 precision. After building, check the reported "Latency" and "Throughput" fields and compare.


.. code-block:: bash

    trtexec --onnx=original.onnx --saveEngine=fp16.engine --fp16


.. note::

    If you replace ``--fp16`` flag with ``--best`` flag, this command will create an int8 engine with TensorRT's implicit quantization.
