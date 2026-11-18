"""Assign persistent object IDs with BoT-SORT on top of the trained detector.

Writes one CSV row per tracked box per frame, and optionally an annotated video.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))  # for legacy_acsh
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.runtime/ultralytics'))

CSV_FIELDS = ['frame', 'source', 'track_id', 'class_id', 'class_name', 'conf',
              'x1', 'y1', 'x2', 'y2', 'foot_x', 'foot_y']


def parse_args():
    parser = argparse.ArgumentParser(description='BoT-SORT 객체 ID 추적')
    parser.add_argument('--weights', required=True,
                        help='학습된 가중치(.pt). 예: runs/detect/acsh_yolov8_train/weights/best.pt')
    parser.add_argument('--source', required=True,
                        help='영상 파일, 이미지 폴더(파일명 순서 = 프레임 순서), 웹캠 번호(0) 또는 스트림 URL')
    parser.add_argument('--tracker', type=Path, default=HERE / 'botsort.yaml')
    parser.add_argument('--conf', type=float, default=0.25, help='검출 신뢰도 하한')
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU 임계값')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default=None, help='cpu 또는 GPU 번호(예: 0). 생략 시 자동')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='추적할 클래스 번호만 지정 (예: --classes 1 → Human만)')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/track',
                        help='결과 상위 폴더')
    parser.add_argument('--name', default='botsort', help='결과 하위 폴더 이름')
    parser.add_argument('--save-video', action='store_true', help='ID가 그려진 영상/이미지 저장')
    parser.add_argument('--show', action='store_true', help='실시간 창으로 표시')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.tracker.is_file():
        raise SystemExit(f'트래커 설정 파일이 없습니다: {args.tracker}')

    from ultralytics import YOLO
    import legacy_acsh

    # 팀 원본 학습 스크립트로 만든 체크포인트(model/best.pt)는 __main__.C2f_ACmix를 찾음
    legacy_acsh.register()
    model = YOLO(args.weights, task='detect')
    names = model.names

    source = int(args.source) if args.source.isdigit() else args.source
    out_dir = (args.out / args.name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'tracks.csv'

    results = model.track(
        source=source, tracker=str(args.tracker.resolve()), persist=True, stream=True,
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
        classes=args.classes, save=args.save_video, show=args.show,
        project=str(out_dir.parent), name=out_dir.name, exist_ok=True, verbose=False,
    )

    ids = set()
    frames = 0
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for frame, result in enumerate(results):
            frames += 1
            boxes = result.boxes
            if boxes is None or boxes.id is None:  # 이번 프레임에 확정된 트랙 없음
                continue
            for (x1, y1, x2, y2), tid, cls, conf in zip(
                    boxes.xyxy.tolist(), boxes.id.int().tolist(),
                    boxes.cls.int().tolist(), boxes.conf.tolist()):
                ids.add(tid)
                writer.writerow({
                    'frame': frame, 'source': Path(result.path).name, 'track_id': tid,
                    'class_id': cls, 'class_name': names[cls], 'conf': round(conf, 4),
                    'x1': round(x1, 1), 'y1': round(y1, 1), 'x2': round(x2, 1), 'y2': round(y2, 1),
                    # 박스 하단 중앙: 바닥 접점. 이후 지면 좌표/히트맵 변환에 사용
                    'foot_x': round((x1 + x2) / 2, 1), 'foot_y': round(y2, 1),
                })

    print(f'처리 프레임: {frames}, 고유 ID 수: {len(ids)}')
    print(f'CSV: {csv_path}')
    if args.save_video:
        print(f'시각화 결과: {out_dir}')


if __name__ == '__main__':
    main()
