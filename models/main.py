import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

def train_yolo():

    DATASET_PATH = "D:/git/Carrot_Detection_harvesting_equipment/data.yaml"     # 여기에 실제 .yaml 경로 입력
    MODEL_NAME = "yolov8n.pt"                                                   # 사전 학습된 모델 or .pt 체크포인트
    RESULT_SAVE_DIR = "D:/git/Carrot_Detection_harvesting_equipment/results"    # 결과 저장할 폴더
    RUN_NAME = "Carrot_run"    # 실험 이름 (모델 저장 경로로 사용됨)

    # 디렉토리 설정
    os.makedirs(RESULT_SAVE_DIR, exist_ok=True)
    PROJECT_DIR = os.path.join(RESULT_SAVE_DIR, RUN_NAME)
    os.makedirs(PROJECT_DIR, exist_ok=True)


    # 모델 로드
    # 새로운 학습이면 yolov8n.pt 등 사용, 재학습이면 best.pt
    model = YOLO(MODEL_NAME)


    # 모델 학습 or 이어서 학습
    resume_training = MODEL_NAME.endswith(".pt") and os.path.exists(MODEL_NAME)

    results = model.train(
        data=DATASET_PATH,
        epochs=30,
        imgsz=640,
        save=True,
        project=RESULT_SAVE_DIR,
        name=RUN_NAME,
        verbose=True,
        plots=True,
        device='0'  # GPU 사용
    )


    # 모델 성능 평가
    metrics = model.val()
    print("\n===== 평가 결과 =====")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")


    # 학습 그래프 저장
    train_metrics_path = os.path.join(PROJECT_DIR, "results.csv")
    if os.path.exists(train_metrics_path):
        df = pd.read_csv(train_metrics_path)

        plt.figure(figsize=(12, 6))
        plt.plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5')
        plt.plot(df['epoch'], df['metrics/precision'], label='Precision')
        plt.plot(df['epoch'], df['metrics/recall'], label='Recall')
        plt.title("YOLOv8 Training Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.grid(True)
        graph_path = os.path.join(PROJECT_DIR, "train_graph.png")
        plt.savefig(graph_path)
        print(f"그래프 저장됨: {graph_path}")
    else:
        print("results.csv 파일을 찾을 수 없습니다. 그래프 생략.")


    # 저장된 모델 불러오기 예시 (best.pt)
    trained_model_path = os.path.join(PROJECT_DIR, "weights", "best.pt")
    if os.path.exists(trained_model_path):
        print(f"저장된 모델 불러오기: {trained_model_path}")
        trained_model = YOLO(trained_model_path)
    else:
        print("best.pt 모델이 존재하지 않습니다.")


if __name__ == '__main__':
    train_yolo()