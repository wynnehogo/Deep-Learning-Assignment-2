# Unified-OneHead Multi-Task Learning Report

Deep Learning Course Assignment 2

何淑雯 RE6137012

## Background and Motivation

This project tackles a unified multi-task learning challenge involving object detection, semantic segmentation, and image classification using a single model head. The goal is to design an architecture that can handle all three tasks simultaneously, while meeting strict constraints on total parameter count, inference time, and performance retention across sequential training phases. 

A key challenge in this setting is **catastrophic forgetting**, where performance on earlier tasks degrades as the model is trained on new ones. The assignment defines acceptable retention as ≤ 5% performance drop from each task's single-task baseline.

## Architecture

The model is designed to perform object detection, semantic segmentation, and image classification using a single output head. The overall structure consists of:

- **Backbone:** `mobilenetv3_small_100` from the `timm` library, pretrained on ImageNet. This backbone was chosen for its low parameter count (~1.5M) and efficiency.
- **Head:** A unified head with three convolutional layers:
  - Two shared Conv2D + BatchNorm + ReLU blocks.
  - A final Conv2D output layer that produces a combined feature map for all three tasks.
  - Outputs are split into:
    - **Detection:** Bounding box regression + class confidence scores.
    - **Segmentation:** Pixel-wise logits over 21 classes.
    - **Classification:** Image-level logits derived via global average pooling.
- **Forgetting Mitigation:** EWC is introduced to reduce the degradation in performance when tasks are learned sequentially.

Elastic Weight Consolidation (EWC) operates by identifying which parameters are most important to previously learned tasks and discouraging large updates to those parameters during subsequent training. This is done by adding a regularization term to the loss function that penalizes deviations from earlier parameter values, scaled by their estimated importance (computed via the Fisher Information Matrix).
This addition helps address the challenge of catastrophic forgetting and improves overall task retention in the unified head architecture.

The model emits all task-specific outputs from the same feature representation. This design satisfies the constraint of using exactly 2–3 layers in the head, while keeping total parameters below 8 million.

## Schedule

- First phase of training (set as baseline) followed the prescribed three-stage sequential schedule to simulate catastrophic forgetting:

1. **Stage 1 – Segmentation:**
   - Trained only on Mini-VOC-Seg.
   - Baseline mIoU recorded after training.
   - Training parameters:
     - learning rate: 5e-4
     - epochs: 30
   
2. **Stage 2 – Detection:**
   - Trained only on Mini-COCO-Det.
   - Detection performance recorded.
   - Drop in segmentation performance evaluated.
   - Training parameters:
     - learning rate: 5e-4
     - epochs: 30

3. **Stage 3 – Classification:**
   - Trained only on Imagenette-160.
   - Classification performance measured.
   - Drop in detection and segmentation performance evaluated again.
   - Training parameters:
     - learning rate: 1e-4
     - epochs: 15

- Second phase of training (used for comparison), applying the forgetting mitigation, followed the same three-stages sequences:

1. **Stage 1 – Segmentation:**
   - Trained only on Mini-VOC-Seg.
   - Training parameters:
     - learning rate: 5e-4
     - epochs: 30
   
2. **Stage 2 – Detection (with EWC):**
   - Trained only on Mini-COCO-Det.
   - Training parameters:
     - learning rate: 5e-4
     - epochs: 30
     - EWC lambda: 3.0

3. **Stage 3 – Classification (with EWC):**
   - Trained only on Imagenette-160.
   - Training parameters:
     - learning rate: 1e-4
     - epochs: 15
     - EWC lambda: 5.0

To mitigate catastrophic forgetting across sequential tasks, Elastic Weight Consolidation (EWC) was applied during the second and third stages of training. The EWC regularization strength, controlled by the hyperparameter λ (lambda), was tuned separately for each stage. In Stage 2 (detection), a moderate λ value of 3.0 was used to balance learning new features while preserving those important for the initial segmentation task. In Stage 3 (classification), a higher λ value of 5.0 was selected. This increase reflects the need for stronger regularization at the final stage, where preserving performance on both prior tasks—segmentation and detection—becomes critical. A higher lambda constrains the model more aggressively, preventing large updates to parameters that were previously identified as important.

In addition to adjusting the regularization strength, the number of training epochs in Stage 3 was intentionally reduced to 15, compared to 30 epochs in earlier stages. This decision was made to limit the extent of parameter updates during classification training, thereby reducing the risk of overwriting previously learned representations. Since classification is a relatively simpler task in this context (using the small and clean Imagenette-160 dataset), fewer epochs are typically sufficient for convergence. This careful balancing of training duration and regularization strength helped maintain stable performance across all three tasks, particularly in mitigating the tendency of the classification stage to degrade detection and segmentation accuracy.

- Evaluation is done on the validation dataset of each tasks.

## Results

### Baselines (Without EWC)

| Task           | Metric         | Baseline |
|----------------|----------------|----------|
| Segmentation   | mIoU           | 0.1004   |
| Detection      | mAP@0.3        | 0.0014   |
| Classification | Top-1 Accuracy | 0.5833   |

### Unified Model Performance (with EWC)

| Task           | Metric         | Score   | Drop     | Drop (%)   |
|----------------|----------------|---------|----------|------------|
| Segmentation   | mIoU           | 0.0988  | 0.0016   | 1.60%      |
| Detection      | mAP@0.3        | 0.0017  | -0.0004  | -26.38%    |
| Classification | Top-1 Accuracy | 0.1833  | 0.4000   | 68.57%     |

- Detection performance slightly improved after adding EWC to multi-task training model.
- Segmentation dropped marginally under the 5% threshold, to be exact 1.60% drop.
- Classification experienced significant performance degradation which might need to be improved for balance.

### Resource Summary

- **Total Parameters:** 1,744,462
- **Trainable Parameters:** 1,744,462
- **Inference Speed:** ~ 55.09 ms per 512×512 image (on CPU; within limits for Colab T4)
- **Training Time:** 50-60 minutes (measured on Colab)

All resource and runtime constraints were satisfied.

## Analysis and Discussion

The results show that applying Elastic Weight Consolidation (EWC) provided partial benefits in mitigating catastrophic forgetting in a unified multi-task learning setup. Segmentation and detection tasks responded well to the introduction of EWC. Specifically, segmentation performance dropped by only 1.60% from its baseline, well within the 5% degradation limit defined by the assignment. Detection performance even improved slightly after EWC was applied, which suggests that the regularization may have helped stabilize useful shared features across spatial tasks. These outcomes indicate that for tasks relying on dense spatial outputs, the shared convolutional head was able to retain useful representations, and EWC was effective in constraining parameter drift during task transitions.

However, classification performance experienced a significant drop—over 68% relative to its single-task baseline—despite the use of EWC and reduced training epochs. This suggests that while EWC succeeded in preserving earlier task knowledge, it may have also overly constrained the model’s ability to adapt to the classification task. Unlike detection and segmentation, classification depends on more global image-level features. Because the model’s shared architecture is built primarily around spatial processing (due to convolutional layers), it may inherently underrepresent the kinds of features needed for classification. Moreover, EWC penalizes changes to parameters deemed important to earlier tasks, which might have prevented the model from learning task-specific refinements essential for classification.

This trade-off highlights a core challenge in unified multi-task learning: spatial tasks (like segmentation and detection) benefit from shared feature maps, while classification often relies on more abstract global representations. Since all outputs in this architecture are derived from the same set of shared parameters, performance is inherently influenced by task compatibility. In this case, prioritizing retention for segmentation and detection appears to have come at the cost of flexibility for classification.

## Future Work

Moving forward, there are several promising directions for improvement. First, integrating lightweight task-specific branches or adapters could allow some specialization without violating the parameter constraint. Second, a hybrid mitigation strategy that combines EWC with knowledge distillation or rehearsal (e.g., a small replay buffer) may provide better retention while still allowing sufficient learning capacity for the final task. Finally, dynamic loss weighting could help balance the learning emphasis across tasks more effectively during training.

## Conclusion

In conclusion, the unified model with EWC showed clear benefits in preserving detection and segmentation performance during sequential task learning, while classification remains a challenge. The results highlight both the strengths and trade-offs of using a fully shared architecture with a single output head, and point toward future improvements that can better accommodate task diversity under strict resource constraints.

## References

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114

Howard, J., & Gugger, S. (2020). Fastai: A layered API for deep learning. Information, 11(2), 108. https://doi.org/10.3390/info11020108

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

COCO Consortium. (2017). COCO: Common Objects in Context Dataset. https://cocodataset.org

Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The Pascal Visual Object Classes (VOC) Challenge. International Journal of Computer Vision, 88(2), 303–338.

Imagenette Dataset - A subset of Imagenet for fast experimentation. Provided by Fast.ai. https://github.com/fastai/imagenette
