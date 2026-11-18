"""Draw track IDs and recent foot-point trails from tracks.csv onto the source video."""

import argparse
import collections
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='tracks.csv를 영상 위에 그려 확인용 영상 생성')
    parser.add_argument('--csv', type=Path, required=True, help='track.py가 만든 tracks.csv')
    parser.add_argument('--source', type=Path, required=True, help='추적에 사용한 원본 영상')
    parser.add_argument('--out', type=Path, default=None, help='출력 경로 (기본: CSV 옆 tracks.mp4)')
    parser.add_argument('--width', type=int, default=1920, help='출력 영상 너비 (비율 유지)')
    parser.add_argument('--trail', type=int, default=30, help='궤적으로 그릴 최근 프레임 수 (0이면 끔)')
    parser.add_argument('--start', type=int, default=0, help='시작 프레임')
    parser.add_argument('--end', type=int, default=None, help='끝 프레임 (포함하지 않음)')
    return parser.parse_args()


def color(track_id):
    """Stable, distinct BGR color per ID."""
    hue = (track_id * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0, 0])


def main():
    args = parse_args()
    rows = collections.defaultdict(list)
    with args.csv.open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows[int(r['frame'])].append(r)

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise SystemExit(f'영상을 열 수 없습니다: {args.source}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    src_w, src_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(args.end or total, total)
    scale = args.width / src_w
    size = (args.width, round(src_h * scale / 2) * 2)

    out = args.out or args.csv.with_name('tracks.mp4')
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    trails = collections.defaultdict(lambda: collections.deque(maxlen=max(args.trail, 1)))
    thick = max(1, round(2 * args.width / 1920))
    for frame in range(args.start, end):
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        for r in rows.get(frame, ()):
            tid = int(r['track_id'])
            c = color(tid)
            x1, y1, x2, y2 = (int(float(r[k]) * scale) for k in ('x1', 'y1', 'x2', 'y2'))
            cv2.rectangle(img, (x1, y1), (x2, y2), c, thick)
            label = f"{tid} {r['class_name']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35 * thick, thick)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35 * thick, (0, 0, 0), thick, cv2.LINE_AA)
            trails[tid].append((frame, int(float(r['foot_x']) * scale), int(float(r['foot_y']) * scale)))
        if args.trail:
            for tid, pts in trails.items():
                recent = [(x, y) for f, x, y in pts if frame - f < args.trail]
                for a, b in zip(recent, recent[1:]):
                    cv2.line(img, a, b, color(tid), thick)
        cv2.putText(img, f'frame {frame}', (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    thick * 0.6, (255, 255, 255), thick + 1, cv2.LINE_AA)
        writer.write(img)
        if frame % 1000 == 0:
            print(f'{frame}/{end}', flush=True)

    writer.release()
    print(f'저장: {out}')


if __name__ == '__main__':
    main()
