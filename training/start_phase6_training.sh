
cd /home/thc1006/dev/music-app/training
source venv_yolo12/bin/activate

nohup python -c "
from ultralytics import YOLO

model = YOLO('/home/thc1006/dev/music-app/training/harmony_omr_v2_phase5/fermata_barline_enhanced/weights/best.pt')

results = model.train(
    data='/home/thc1006/dev/music-app/training/datasets/yolo_harmony_v2_phase6_ultimate/harmony_phase6_ultimate.yaml',
    epochs=200,
    batch=16,
    imgsz=640,
    device=0,
    project='/home/thc1006/dev/music-app/training/harmony_omr_v2_phase6',
    name='ultimate_barline_fixed',
    box=7.5,
    cls=2.5,
    dfl=1.5,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.0,
    degrees=0.0,
    flipud=0.0,
    fliplr=0.0,
    patience=50,
    save_period=10,
    verbose=True,
)

print('Training completed!')
best_map = results.results_dict.get('metrics/mAP50(B)', 'N/A')
print('Best mAP50: ' + str(best_map))
" > /home/thc1006/dev/music-app/training/phase6_training.log 2>&1 &

echo $! > /home/thc1006/dev/music-app/training/phase6_training.pid
echo "Training started with PID: $(cat /home/thc1006/dev/music-app/training/phase6_training.pid)"
