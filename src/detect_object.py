import cv2
import os
import sys
from main import ACmix, C2f_ACmix
from ultralytics import YOLO

main_module = sys.modules["__main__"]

setattr(main_module, "ACmix", ACmix)
setattr(main_module, "C2f_ACmix", C2f_ACmix)


# 데이터 경로
MODEL_PATH = 'best.pt'
INPUT_PATH = 'VID1.mp4'
OUTPUT_PATH = 'result.mp4'
ONNX_PATH = MODEL_PATH.replace('.pt', '_int8.onnx')

if not os.path.exists(ONNX_PATH):
    temp_model = YOLO(MODEL_PATH)
    temp_model.export(format="onnx", imgsz=640, quantize="int8", data="data.yaml", device=0)
model = YOLO(ONNX_PATH, task='detect')


print("모델 로드 완료")
print(f"입력 파일: {INPUT_PATH}")


# 파일 종류 확인
image_extensions = [
    '.jpg', '.jpeg', '.png', '.bmp', '.webp'
]

video_extensions = [
    '.mp4', '.avi', '.mov', '.mkv', '.wmv'
]

extension = os.path.splitext(INPUT_PATH)[1].lower()

IMG_SIZE = 640
CONF = 0.15
DEVICE = 0

# 이미지 처리
if extension in image_extensions:

    image = cv2.imread(INPUT_PATH)

    if image is None:
        print("이미지를 불러올 수 없습니다.")
        exit()

    # YOLO 추론
    results = model.predict(
        source=image,
        imgsz=IMG_SIZE,
        conf=CONF,
        device=DEVICE,
        verbose=False
    )

    # 결과 표시
    annotated_frame = results[0].plot()

    cv2.imshow("Car and Human Detection", annotated_frame)

    print("이미지 결과입니다.")
    print("아무 키나 누르면 종료합니다.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 영상 처리
elif extension in video_extensions:

    cap = cv2.VideoCapture(INPUT_PATH)

    if not cap.isOpened():
        print("영상을 열 수 없습니다.")
        exit()

    # 영상 정보
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"영상 크기: {width} x {height}")
    print(f"FPS: {fps}")

    # 결과 영상 저장
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height)
    )

    while True:

        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            print("영상 처리가 완료되었습니다.")
            break

        # YOLO 추론
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF,
            device=DEVICE,
            verbose=False
        )

        # 탐지 결과
        annotated_frame = results[0].plot()

        # 화면 출력
        cv2.imshow("Car and Human Detection", annotated_frame)

        # 결과 영상 저장
        out.write(annotated_frame)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자가 종료했습니다.")
            break

    # 종료
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"결과 영상 저장 완료: {OUTPUT_PATH}")

else:
    print("지원하지 않는 파일 형식입니다.")
    print("이미지: jpg, jpeg, png, bmp, webp")
    print("영상: mp4, avi, mov, mkv, wmv")