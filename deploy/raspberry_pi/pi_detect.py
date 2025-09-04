#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
라즈베리파이 실사용 스크립트
- 카메라 프레임 캡처
- YOLOv8 ONNX 추론 (동적/정적 헤드 자동 파싱)
- ROI/면적/연속프레임 기준으로 컷 신호 생성
- 아두이노로 "CUT\n" / "IDLE\n" 송신
"""

import argparse
import sys, time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

try:
    import serial
except Exception:
    serial = None  # 시리얼 미사용 모드 허용


# -------------------------
# 유틸
# -------------------------
def letterbox(img, new_size, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = new_size - nh, new_size - nw
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top), (nw, nh)


def xywh2xyxy(box):
    # box: (cx, cy, w, h)
    cx, cy, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def nms_cv2(boxes_xyxy, scores, iou_thres):
    if len(boxes_xyxy) == 0:
        return []
    # OpenCV NMS expects [x,y,w,h]
    wh_boxes = boxes_xyxy.copy()
    wh_boxes[:, 2] = wh_boxes[:, 2] - wh_boxes[:, 0]
    wh_boxes[:, 3] = wh_boxes[:, 3] - wh_boxes[:, 1]
    idxs = cv2.dnn.NMSBoxes(
        bboxes=wh_boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=float(iou_thres)
    )
    return [i[0] for i in idxs] if len(idxs) else []


def parse_yolov8_output(out, num_classes):
    """
    지원 형태:
      1) (1, N, 5+K)   # 정적 헤드
      2) (1, 5+K, N)   # 동적 헤드(일반적 export)
    반환: boxes(xywh), objectness, class_scores  -> 모두 (N, ...)
    """
    if out.ndim == 3:
        b, a, c = out.shape
        if a == (5 + num_classes):  # (1, 5+K, N)
            out = np.transpose(out, (0, 2, 1))  # -> (1, N, 5+K)
        elif c == (5 + num_classes):  # (1, N, 5+K)
            pass
        else:
            raise RuntimeError(f"알 수 없는 ONNX 출력 형태: {out.shape}")
        out = out[0]  # (N, 5+K)
    elif out.ndim == 2:  # (N, 5+K)
        pass
    else:
        raise RuntimeError(f"알 수 없는 ONNX 출력 차원: {out.shape}")

    boxes_xywh = out[:, 0:4]
    obj = out[:, 4]
    cls_scores = out[:, 5:]
    return boxes_xywh, obj, cls_scores


# -------------------------
# 메인
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="ONNX 경로")
    p.add_argument("--device", type=int, default=0, help="카메라 인덱스")
    p.add_argument("--serial", type=str, default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.50)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--class", dest="cls_id", type=int, default=0, help="당근 클래스 인덱스")
    p.add_argument("--roi", type=str, default="0.40,1.00", help="세로 ROI: ymin,ymax (0~1)")
    p.add_argument("--area", type=float, default=0.060, help="면적 임계(프레임 대비)")
    p.add_argument("--hits", type=int, default=3, help="연속 히트 프레임")
    args = p.parse_args()

    ymin, ymax = [float(x) for x in args.roi.split(",")]
    assert 0.0 <= ymin < ymax <= 1.0

    # 시리얼
    ser = None
    if serial is not None:
        try:
            ser = serial.Serial(args.serial, args.baud, timeout=0.05)
            time.sleep(2.0)
        except Exception as e:
            print(f"[경고] 시리얼 포트 열기 실패: {e}. (시리얼 없이 계속 진행)", file=sys.stderr)

    # ONNX 세션
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # 클래스 수 추정 (입력 텐서 shape로 판단 불가 → 첫 추론 후 자동 도출)
    # 일단 카메라 오픈
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    hit_queue = deque(maxlen=args.hits)
    num_classes_cached = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        blob, r, (dx, dy), (nw, nh) = letterbox(frame, args.imgsz)

        x = blob[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)

        # 추론
        preds = sess.run([out_name], {in_name: x})[0]

        # 클래스 수 자동 결정 (1회)
        if num_classes_cached is None:
            if preds.ndim == 3:
                a = preds.shape[1] if preds.shape[1] <= preds.shape[2] else preds.shape[2]
                # a == 5 + K
                if a < 6:
                    raise RuntimeError(f"출력 차원 파악 실패: {preds.shape}")
                num_classes_cached = a - 5
            elif preds.ndim == 2:
                a = preds.shape[1]  # (N, 5+K)
                if a < 6:
                    raise RuntimeError(f"출력 차원 파악 실패: {preds.shape}")
                num_classes_cached = a - 5
            else:
                raise RuntimeError(f"알 수 없는 출력: {preds.shape}")

        boxes_xywh, obj, cls_scores = parse_yolov8_output(preds, num_classes_cached)

        # 점수 계산 (obj * class_prob)
        cls_id = args.cls_id
        if cls_id >= num_classes_cached:
            raise RuntimeError(f"--class {cls_id} 가 모델 클래스수 {num_classes_cached} 를 초과")

        scores = obj * cls_scores[:, cls_id]
        keep_mask = scores >= args.conf

        boxes_xywh = boxes_xywh[keep_mask]
        scores = scores[keep_mask]

        # (cx,cy,w,h) → (x1,y1,x2,y2) (letterbox 좌표계)
        boxes_xyxy = xywh2xyxy(boxes_xywh)

        # letterbox 역보정 → 원본 좌표계
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - dx) / r
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - dy) / r

        # 클리핑
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, W)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, H)

        # 유효 박스만 남기기 (너비/높이 > 0)
        wh = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        valid = wh > 0
        boxes_xyxy = boxes_xyxy[valid]
        scores = scores[valid]

        # NMS
        keep_idx = nms_cv2(boxes_xyxy.copy(), scores.copy(), args.iou)
        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = scores[keep_idx]

        # 트리거 판정
        trigger = False
        for (x1, y1, x2, y2), sc in zip(boxes_xyxy, scores):
            cx = (y1 + y2) / 2.0
            if cx < H * ymin or cx > H * ymax:
                continue
            area_ratio = ((x2 - x1) * (y2 - y1)) / float(W * H)
            if area_ratio < args.area:
                continue
            trigger = True

            # 시각화
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(frame, f"carrot {sc:.2f}", (int(x1), max(15, int(y1)-7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        hit_queue.append(1 if trigger else 0)
        if sum(hit_queue) >= args.hits:
            if ser:
                try:
                    ser.write(b"CUT\n")
                except Exception:
                    pass
            cv2.putText(frame, "CUT!", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            hit_queue.clear()
        else:
            if ser:
                try:
                    ser.write(b"IDLE\n")
                except Exception:
                    pass

        # ROI 라인 표시
        cv2.line(frame, (0, int(H * ymin)), (W, int(H * ymin)), (255, 255, 0), 1)
        cv2.line(frame, (0, int(H * ymax)), (W, int(H * ymax)), (255, 255, 0), 1)

        cv2.imshow("carrot-det", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
