# BoT-SORT 객체 ID 추적

학습된 사람·차량 탐지 모델(ACSH-YOLOv8)의 검출 결과에 **BoT-SORT**를 적용해 프레임 사이에서 같은 객체에 같은 ID를 붙입니다. 결과는 프레임별 CSV로 저장되어 이후 밀집도 계산이나 3D 히트맵 단계의 입력으로 쓸 수 있습니다.

이 폴더는 다른 코드에 의존하지 않고 단독으로 동작합니다. 저장소의 `model/best.pt`(팀 학습 모델)를 바로 사용할 수 있습니다.

## 구성

```text
tracking/
├── track.py          # 실행 스크립트
├── legacy_acsh.py    # model/best.pt 로딩용 ACmix/C2f_ACmix 정의
├── render.py         # tracks.csv를 영상 위에 그려 확인용 영상 생성
├── botsort.yaml      # BoT-SORT 하이퍼파라미터
├── requirements.txt  # 의존성 (torch, ultralytics, lap)
└── README.md
```

## 설치

Python 3.12 기준입니다. 저장소 루트에서 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r tracking/requirements.txt
```

`lap`은 Ultralytics 트래커가 헝가리안 매칭에 사용하는 패키지입니다. 설치하지 않으면 `ModuleNotFoundError: No module named 'lap'`이 납니다.

## 사용법

프로젝트 루트에서 실행합니다.

```bash
# 팀 학습 모델(model/best.pt)로 영상 추적
python tracking/track.py \
    --weights model/best.pt \
    --source path/to/video.mp4 \
    --save-video

# 사람(Human = 1)만 추적
python tracking/track.py --weights model/best.pt --source video.mp4 --classes 1

# 연속 프레임 이미지 폴더 (파일명 정렬 순서 = 프레임 순서)
python tracking/track.py --weights model/best.pt --source path/to/frames/

# 웹캠 / RTSP 스트림
python tracking/track.py --weights model/best.pt --source 0 --show
python tracking/track.py --weights model/best.pt --source rtsp://... --show
```

학습된 가중치가 아직 없을 때는 COCO 사전학습 모델로 동작만 확인할 수 있습니다. 이 경우 클래스 번호가 다르므로(0: person, 2: car) `--classes 0 2`를 사용합니다. `yolov8n.pt`는 처음 실행할 때 자동으로 내려받습니다.

```bash
python tracking/track.py --weights yolov8n.pt --source video.mp4 --classes 0 2 --save-video
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--weights` | (필수) | 학습된 `.pt` 경로 |
| `--source` | (필수) | 영상 파일, 이미지 폴더, 웹캠 번호, 스트림 URL |
| `--tracker` | `tracking/botsort.yaml` | 트래커 설정 파일 |
| `--conf` | 0.25 | 검출 신뢰도 하한 |
| `--iou` | 0.7 | NMS IoU 임계값 |
| `--imgsz` | 640 | 추론 입력 크기 |
| `--device` | 자동 | `cpu` 또는 GPU 번호(`0`) |
| `--classes` | 전체 | 추적할 클래스 번호 |
| `--out`, `--name` | `runs/track`, `botsort` | 결과 저장 위치 → `runs/track/botsort/` |
| `--save-video` | 꺼짐 | ID가 그려진 영상/이미지 저장 |
| `--show` | 꺼짐 | 실시간 창 표시 |

## 출력

`runs/track/<name>/tracks.csv`에는 확정된 트랙 박스가 한 행씩 저장됩니다.

| 열 | 의미 |
| --- | --- |
| `frame` | 0부터 시작하는 프레임 번호 |
| `source` | 입력 파일 이름 (이미지 폴더일 때 프레임 파일명) |
| `track_id` | BoT-SORT가 부여한 객체 ID |
| `class_id`, `class_name` | 클래스 (`0: Car`, `1: Human`) |
| `conf` | 검출 신뢰도 |
| `x1, y1, x2, y2` | 박스 좌표(픽셀, 원본 이미지 기준) |
| `foot_x, foot_y` | 박스 하단 중앙. 객체가 바닥에 닿는 지점이며, 이후 카메라 보정을 거쳐 지면 좌표로 바꿀 때 사용합니다 |

`--save-video`를 켜면 같은 폴더에 ID가 그려진 영상 또는 이미지가 저장됩니다. 다만 원본 해상도 그대로 저장되므로, 5.7K 영상이면 파일이 매우 커집니다. 이럴 때는 아래 `render.py`를 쓰세요.

## 확인용 영상 만들기 (`render.py`)

추적이 끝난 뒤 `tracks.csv`를 원본 영상 위에 그립니다. 추적을 다시 돌리지 않으므로, 해상도나 구간을 바꿔 여러 번 만들 수 있습니다. ID별로 고정된 색의 박스와 라벨(`ID 클래스`)을 그리고, 최근 프레임 동안 이동한 발 위치 궤적을 선으로 표시합니다.

```bash
python tracking/render.py \
    --csv runs/track/botsort/tracks.csv \
    --source path/to/video.mp4            # 결과: CSV 옆 tracks.mp4 (1920px 너비)

# 일부 구간만, 더 작게
python tracking/render.py --csv ... --source ... --start 900 --end 1800 --width 1280 --out clip.mp4
```

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--width` | 1920 | 출력 너비 (비율 유지) |
| `--trail` | 30 | 궤적으로 그릴 최근 프레임 수 (0이면 끔) |
| `--start`, `--end` | 전체 | 프레임 구간 |
| `--out` | CSV 옆 `tracks.mp4` | 출력 경로 |

출력 코덱은 OpenCV 기본 `mp4v`입니다. VLC나 기본 동영상 플레이어에서는 재생되지만, 웹 브라우저에서는 재생되지 않을 수 있습니다.

## BoT-SORT 설정 (`botsort.yaml`)

Ultralytics 8.3.0의 기본값과 같습니다. 자주 조정하는 값은 다음과 같습니다.

- `track_buffer`: 가려지거나 놓친 객체의 ID를 유지하는 프레임 수입니다. 이 버전은 영상 FPS와 관계없이 30fps로 가정하므로, 30이면 약 1초입니다. 사람이 자주 가려지면 늘리세요.
- `new_track_thresh`: 이 값보다 신뢰도가 높아야 새 ID를 만듭니다. 오검출로 ID가 남발되면 올리세요.
- `track_high_thresh` / `track_low_thresh`: 두 단계 매칭에 쓰는 신뢰도 기준입니다.
- `gmc_method`: 카메라 움직임 보정 방식입니다. 고정 CCTV라면 `None`으로 두면 더 빠르고, 차량에 달린 카메라처럼 움직이는 영상은 `sparseOptFlow`(기본)나 `ecc`를 쓰세요.
- `with_reid`: Ultralytics 8.3.0은 외형 특징(ReID) 매칭을 지원하지 않습니다. 따라서 ID 연결은 위치·움직임(칼만 필터)과 IoU에만 의존합니다.

ByteTrack과 비교하려면 `tracker_type: bytetrack`으로 바꾸거나 Ultralytics 내장 `bytetrack.yaml`을 `--tracker`로 넘기면 됩니다.

## 주의사항

- **`model/best.pt` 로딩**: 이 체크포인트는 원본 학습 스크립트 안에 정의된 `C2f_ACmix`를 `__main__.C2f_ACmix`라는 이름으로 저장하고 있습니다. 그래서 그냥 `YOLO('model/best.pt')`를 하면 `AttributeError: Can't get attribute 'C2f_ACmix' on <module '__main__'>`가 납니다. `track.py`는 `legacy_acsh.register()`로 이 문제를 처리합니다. 다른 스크립트에서 이 가중치를 쓸 때도 먼저 `legacy_acsh.register()`를 호출하세요.

- **입력은 연속된 원본 영상이어야 합니다.** `detection human and car/train` 이미지는 Roboflow 증강(4장 모자이크, 검은 박스 가림, 프레임당 여러 사본)을 거친 학습용 데이터라 프레임 사이에 객체가 이어지지 않습니다. 이 이미지로는 추적 품질을 평가할 수 없습니다. 원본 촬영 영상을 사용하세요.
- 이미지 폴더를 입력하면 하나의 연속 시퀀스로 처리합니다. 여러 영상의 프레임을 한 폴더에 섞지 마세요.
- ID는 실행할 때마다 1부터 다시 매겨집니다. 서로 다른 영상이나 카메라 사이에서는 같은 객체라도 같은 ID가 보장되지 않습니다.
- 새 트랙은 첫 프레임을 제외하면 두 번 연속 매칭되어야 확정되고 CSV에 기록됩니다. 그래서 한 프레임만 나타난 객체는 기록되지 않습니다.
- **360° 등장방형 영상의 좌우 경계**: 영상의 왼쪽 끝과 오른쪽 끝은 실제로는 이어진 같은 방향(차량 뒤쪽)입니다. 하지만 트래커는 이를 모르므로, 경계를 넘어가는 객체는 새 ID를 받습니다.
- 원본이 5.7K(5760×2880)이면 디코딩이 느립니다. 1920×960 정도로 줄인 영상을 `--imgsz 1280`으로 처리하는 것을 권장합니다.

## 검증 상태

- COCO `yolov8n.pt`와 train 이미지 85장으로 실행해, 검출 → ID 부여 → CSV·시각화 저장까지 오류 없이 동작하는 것을 확인했습니다.
- 실제 주행 영상 `VID_20250820_085852_00_265-032-001.mp4`의 앞 60초를 1920×960으로 줄여 실행했습니다(`yolov8n.pt`, `--imgsz 1280`, `--classes 0 2`, GPU).
  - 결과: 1798프레임을 44초에 처리(약 41fps), 고유 ID 162개(차량 161, 사람 1), 트랙 길이 중앙값 70프레임(약 2.3초).
  - 끊겼다가 같은 ID로 다시 이어진 트랙이 39개입니다.
  - 1초 간격 두 프레임을 눈으로 비교했을 때 앞차, 맞은편 차, 추월 차량, 주차 차량의 ID가 유지되었습니다.
- 사람 ID가 1개뿐인 이유는 COCO 소형 모델이 360° 영상의 작은 보행자를 잘 잡지 못하기 때문으로 보입니다.
- 같은 60초 구간을 팀 학습 모델 `model/best.pt`로 다시 실행했습니다(`--imgsz 1280`).
  - 결과: 고유 ID 638개(모두 Car), 프레임당 박스 24.7개, 트랙 길이 중앙값 22프레임, 30프레임 이하 짧은 트랙 355개(56%).
  - COCO 모델보다 차량을 2.5배 많이 검출했지만, 짧게 끊기는 트랙이 많습니다. Human은 한 번도 검출되지 않았습니다(학습 데이터의 Human 객체 수가 적은 영향으로 보입니다).
- 정답 트랙 라벨이 없어서 IDF1이나 ID switch 같은 정량 지표는 아직 계산하지 않았습니다.
