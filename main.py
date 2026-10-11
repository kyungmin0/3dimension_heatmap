import os
import sys
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Conv

verbose = False
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ACmix, C2f_ACmix 클래스 정의
class ACmix(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        self.cv1 = (
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False)
            if c1 != c2
            else nn.Identity()
        )

        # Q, K, V
        self.qkv_conv = nn.Conv2d(
            c2,
            c2 * 3,
            kernel_size=1,
            stride=1,
            bias=False
        )

        # Local convolution branch
        self.conv_branch = nn.Conv2d(
            c2,
            c2,
            kernel_size=k,
            stride=s,
            padding=k // 2,
            groups=c2,
            bias=False
        )

        # Global context
        self.pool = nn.AdaptiveAvgPool2d(1)

        hidden = max(c2 // 4, 8)

        self.fc1 = nn.Conv2d(
            c2,
            hidden,
            kernel_size=1
        )

        self.fc2 = nn.Conv2d(
            hidden,
            c2,
            kernel_size=1
        )

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        self.act = nn.SiLU()

    def forward(self, x):

        x = self.cv1(x)

        # Q / K / V
        qkv = self.qkv_conv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        f_conv = self.conv_branch(v)

       
        attn = self.pool(q * k)

        attn = self.fc1(attn)
        attn = self.act(attn)

        attn = self.fc2(attn)
        attn = torch.sigmoid(attn)

        f_att = v * attn

        out = self.alpha * f_conv + self.beta * f_att

        return self.act(out)


class C2f_ACmix(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        g=1,
        e=0.5
    ):
        super().__init__()

        # C2f
        self.c2f = C2f(
            c1,
            c2,
            n=n,
            shortcut=shortcut,
            g=g,
            e=e
        )

        # ACmix
        self.acmix = ACmix(c2, c2)

    def forward(self, x):

        x = self.c2f(x)
        x = self.acmix(x)
        return x

# Ultralytics 라이브러리 동적 등록 (Monkey Patching)
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules

setattr(tasks, 'C2f_ACmix', C2f_ACmix)
setattr(modules, 'C2f_ACmix', C2f_ACmix)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"현재 사용 장치: {device}")
    
    if device == 'cuda':
        print(f"GPU 모델명: {torch.cuda.get_device_name(0)}")

    # ACSH-YOLOv8 모델 로드
    print("\nACSH-YOLOv8 모델 로드...")
    model = YOLO(r'C:\Users\chsr1\project\pothole\acsh-yolov8.yaml')

    model.info(verbose=True)

    # 모델 학습
    print("\n모델 학습 시작")
    
    results = model.train(
        data=r'C:\Users\chsr1\project\pothole\data.yaml',  
        epochs=200,
        imgsz=640,                  
        batch=16,
        
        # --- 데이터 증강 (Data Augmentation) ---
        conf=0.2,
        hsv_v=0.2,        # 명도 변환
        degrees=5,        # 회전
        scale=0.3,        # 확대/축소
        fliplr=0.5,       # 좌우 뒤집기
        mosaic=0.5,       # 모자이크 증강
        
        # --- 옵티마이저 & 학습률 ---
        lr0=0.001,
        lrf=0.01,
        optimizer='AdamW',

        # --- 학습 전략 ---
        close_mosaic=20,  # 마지막 20 에폭에서는 모자이크 증강 비활성화
        patience=30,      # 성능 개선 없으면 조기 종료

        cache=False,
        device=0,      
        workers=4,                     
        name='acsh_yolov8_train'
    )

    print("\n모델 학습 완료")

# 에러 방지 코드
if __name__ == '__main__':
    main()