# 示例 1: 从 'my_video.mp4' 中每秒提取 1 帧，保存到 'output_frames' 目录
python frame_extractor.py -i "my_video.mp4" -o "output_frames" --fps 1

# 示例 2: 每 2 秒提取 1 帧 (fps=0.5)，使用 'shot_' 前缀，并保存为 png
python frame_extractor.py -i "Bellevue_150th_Eastgate_clip.mp4" \
                        -o "bellevue_shots" \
                        --fps 0.5 \
                        --prefix "shot_" \
                        --format "png"

# 示例 3: 提取 10 FPS (如果源视频是 30fps，则每 3 帧保存一帧)
python frame_extractor.py -i "my_video.mp4" -o "high_fps_frames" --fps 10