# Knowledge Distillation for Traffic Sign Recognition

This project evaluates knowledge distillation as a model-compression technique for traffic sign recognition. A MobileNetV2 teacher transfers its output distribution to a compact, depthwise-separable CNN trained on the German Traffic Sign Recognition Benchmark (GTSRB). The experiments compare the teacher, a student trained with hard labels, and nine distilled students across accuracy, parameter count, inference latency, and class-level errors.

The complete experiment is available in [`gtsrb.ipynb`](./gtsrb.ipynb) and as a [Kaggle notebook](https://www.kaggle.com/code/dongmyungpark/gtsrb).

## Results

| Model | Parameters | Accuracy | Latency (batch size 1) |
| --- | ---: | ---: | ---: |
| MobileNetV2 teacher | 2,313,067 | 88.37% | Not measured |
| Student baseline | 34,406 | 54.61% | 54.01 ± 2.96 ms/image |
| Distilled student (`T=10`, `alpha=0.3`) | 34,406 | 80.97% | 55.19 ± 3.79 ms/image |

The best distilled student improves on the baseline by 26.36 percentage points while retaining the same architecture and parameter count. It reaches 91.6% of the teacher's accuracy with approximately 67 times fewer parameters. Across the nine distillation settings, measured latency ranged from 53.68 to 55.19 ms per image.

The class-level analysis found a larger mean improvement for speed-limit signs (classes 0-8) than for the remaining classes: +33.11 percentage points compared with +23.44 percentage points. Some directional-sign classes regressed, indicating that visually similar classes do not always provide a beneficial distillation signal.

All values above are taken from the saved notebook outputs. The notebook uses the official GTSRB test split as `val_ds` during model selection, so these figures should be treated as experimental comparisons rather than estimates from an untouched final test set. Latency measurements are specific to the recorded Kaggle environment and include framework prediction overhead.

## Method

### Dataset and preprocessing

- GTSRB: 43 traffic-sign classes
- 39,209 training images and 12,630 test images
- Images resized to 96 × 96 and normalized to `[0, 1]`
- Inverse-frequency class weights used to address class imbalance
- Fixed random seed (`10879360`) used throughout the experiments

### Models

The teacher uses an ImageNet-pretrained MobileNetV2 backbone. The final 60 backbone layers are fine-tuned, followed by global average pooling, dropout, and a 43-class output layer.

The student is a 34,406-parameter CNN composed of three depthwise-separable convolution blocks, batch normalization, pooling, global average pooling, and a small dense classifier. The same student architecture is used for the baseline and every distillation run.

### Distillation

The student loss combines sparse categorical cross-entropy on the ground-truth labels with KL divergence between the temperature-scaled teacher and student predictions:

```text
loss = alpha * hard_loss
     + (1 - alpha) * temperature^2 * soft_loss
```

The notebook evaluates a 3 × 3 grid of hyperparameters:

- Temperature: `3`, `5`, `10`
- Hard-label weight (`alpha`): `0.1`, `0.3`, `0.5`

The best recorded configuration is `temperature=10` and `alpha=0.3`.

## Running the notebook

The notebook was executed with Python 3.12 and TensorFlow 2.19 on a Kaggle GPU runtime. Kaggle is the simplest way to reproduce the experiment because the notebook downloads the dataset through `kagglehub` and the saved metadata already references a GPU-enabled environment.

To run it locally, create an environment and install the required packages:

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install tensorflow==2.19.0 kagglehub numpy pandas matplotlib scikit-learn jupyter
jupyter notebook gtsrb.ipynb
```

Local execution requires internet access for the GTSRB dataset and the pretrained MobileNetV2 weights. A CUDA-capable GPU is recommended; the recorded end-to-end notebook run took approximately 56 minutes on a Tesla P100.

## Repository contents

| File | Description |
| --- | --- |
| [`gtsrb.ipynb`](./gtsrb.ipynb) | Data preparation, model training, hyperparameter search, latency measurement, and error analysis |
| [`34212-Lab-S-Report.pdf`](./34212-Lab-S-Report.pdf) | Five-page coursework report covering the robotics context, methodology, and findings |
| [`COMP34212_Coursework_2026.pdf`](./COMP34212_Coursework_2026.pdf) | Coursework specification and marking criteria |

Running the notebook also produces comparison and confusion-matrix figures in the current working directory.
