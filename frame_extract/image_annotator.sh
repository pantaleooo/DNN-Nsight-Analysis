# 示例 1: 使用默认的 FasterR-CNN 模型，在 GPU 上运行 (如果可用)，保存到 'annotated_frames'
python image_annotator.py -i "output_frames" -o "annotated_frames"

# 示例 2: 使用 RetinaNet 模型，提高置信度门槛，并强制使用 CPU
python image_annotator.py -i "output_frames" \
                        -o "annotated_frames_retinanet" \
                        --model_name "retinanet_resnet50_fpn_v2" \
                        --threshold 0.7 \
                        --device "cpu"