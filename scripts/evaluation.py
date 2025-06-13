# Prediction Decoder
def decode_predictions(det_out, conf_thresh=0.01, S=7, num_classes=10):
    B, _, H, W = det_out.shape
    assert H == W == S
    pred = det_out.permute(0, 2, 3, 1)  # [B, S, S, 5+C]
    decoded = []

    for b in range(B):
        boxes = []
        scores = []
        labels = []
        for i in range(S):
            for j in range(S):
                cell = pred[b, i, j]
                cx, cy, w, h = cell[:4]
                obj_logit = cell[4]
                class_scores = F.softmax(cell[5:], dim=0)

                # Apply activations
                obj_conf = torch.sigmoid(obj_logit)
                cx = torch.sigmoid(cx)
                cy = torch.sigmoid(cy)
                w = torch.exp(w) if w < 10 else torch.tensor(1.0)  # safety
                h = torch.exp(h) if h < 10 else torch.tensor(1.0)

                # Compute box center in image space
                cx_abs = (j + cx.item()) / S * 224
                cy_abs = (i + cy.item()) / S * 224
                w_abs = w.item() * 224
                h_abs = h.item() * 224

                x1 = cx_abs - w_abs / 2
                y1 = cy_abs - h_abs / 2
                x2 = cx_abs + w_abs / 2
                y2 = cy_abs + h_abs / 2

                score, cls = class_scores.max(0)
                final_score = (score * obj_conf).item()

                if final_score > conf_thresh:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(final_score)
                    labels.append(cls.item())

        decoded.append({
            'boxes': torch.tensor(boxes),
            'scores': torch.tensor(scores),
            'labels': torch.tensor(labels)
        })

    return decoded

# IoU Computation
def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0.0
def evaluate_map(preds, targets, iou_thresh=0.3):
    all_true = 0
    all_pred = 0
    true_positives = 0

    for pred, target in zip(preds, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        gt_boxes = target['boxes']
        gt_labels = target['labels']

        matched = set()
        for i, p_box in enumerate(pred_boxes):
            p_label = pred_labels[i]
            found_match = False
            for j, gt_box in enumerate(gt_boxes):
                if j in matched:
                    continue
                if p_label != gt_labels[j]:
                    continue
                iou = compute_iou(p_box.tolist(), gt_box.tolist())
                if iou >= iou_thresh:
                    true_positives += 1
                    matched.add(j)
                    found_match = True
                    break
        all_true += len(gt_boxes)
        all_pred += len(pred_boxes)

    precision = true_positives / all_pred if all_pred > 0 else 0
    recall = true_positives / all_true if all_true > 0 else 0
    return precision, recall, precision  

# Classification Evaluation Function
def evaluate_classification(model_path, val_loader, device):
    model = UnifiedModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, _, cls_out = model(x)
            preds.append(cls_out.argmax(1).cpu())
            labels.append(y.cpu())

    acc = accuracy_score(torch.cat(labels), torch.cat(preds))
    print(f"[{os.path.basename(model_path)}] Top-1 Accuracy: {acc:.4f}")
    return acc
# Segmentation Evaluation Function
def evaluate_segmentation(model_path, val_loader, device):
    model = UnifiedModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_miou = 0
    count = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            seg_out, _, _ = model(x)
            seg_out = F.interpolate(seg_out, size=y.shape[1:], mode='bilinear', align_corners=False)
            total_miou += compute_miou(seg_out.cpu(), y.cpu())
            count += 1
    miou = total_miou / count
    print(f"[{os.path.basename(model_path)}] mIoU: {miou:.4f}")
    return miou
# Detection Evaluation Function
def evaluate_detection_map(model_path, dataloader, device):
    model = UnifiedModel(num_classes=10, seg_classes=21, det_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            _, det_out, _ = model(imgs)
            preds = decode_predictions(det_out.cpu())
            all_preds.extend(preds)
            all_targets.extend(targets)

    precision, recall, approx_map = evaluate_map(all_preds, all_targets)
    print(f"[{os.path.basename(model_path)}] mAP@0.3 (approx): {approx_map:.4f}")
    return approx_map

# Classification
acc_base = evaluate_classification("baseline_cls.pth", val_loader_cls, device)
acc_final = evaluate_classification("unified_model.pth", val_loader_cls, device)
acc_drop = acc_base - acc_final
print(f"Top-1 drop: {acc_drop:.4f} ({(acc_drop / acc_base) * 100:.2f}%)")

# Segmentation
miou_base = evaluate_segmentation("baseline_seg.pth", val_loader_seg, device)
miou_final = evaluate_segmentation("unified_model.pth", val_loader_seg, device)
miou_drop = miou_base - miou_final
print(f"mIoU drop: {miou_drop:.4f} ({(miou_drop / miou_base) * 100:.2f}%)")

# Detection
baseline_map = evaluate_detection_map("baseline_det.pth", val_loader_det, device)
final_map = evaluate_detection_map("unified_model.pth", val_loader_det, device)

drop = baseline_map - final_map
print(f"mAP drop: {drop:.4f} ({(drop / baseline_map) * 100:.2f}%)")
