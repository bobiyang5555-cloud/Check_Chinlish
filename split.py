import argparse
import shutil
import subprocess
import sys
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".webm", ".m4v"}


def find_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    raise FileNotFoundError(
        "未找到 ffmpeg。请先安装 ffmpeg，或运行: python -m pip install imageio-ffmpeg"
    )


def collect_videos(inputs):
    videos = []
    for item in inputs:
        p = Path(item)
        if not p.exists():
            print(f"[跳过] 路径不存在: {p}")
            continue

        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                    videos.append(f)

    # 去重并排序
    return sorted(list({v.resolve() for v in videos}))


def clear_pngs(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for png in output_dir.glob("*.png"):
        png.unlink()


def run_cmd(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def run_ffmpeg(ffmpeg, input_video, output_dir, dedupe=False):
    clear_pngs(output_dir)

    output_pattern = str(output_dir / "%06d.png")
    cmd = [ffmpeg, "-hide_banner", "-y", "-i", str(input_video)]

    # 默认：导出每一帧“显示帧”
    # -vsync 0：尽量按原视频帧逐帧输出，不强制补帧/丢帧
    if dedupe:
        # 去除连续重复/近重复帧（视觉上几乎相同的帧）
        # mpdecimate：删除重复帧
        # setpts=N/FRAME_RATE/TB：重新整理时间戳，避免输出编号异常
        cmd += ["-vf", "mpdecimate,setpts=N/FRAME_RATE/TB", "-vsync", "vfr"]
    else:
        cmd += ["-vsync", "0"]

    cmd += [output_pattern]

    print(f"[开始] {input_video.name}")
    result = run_cmd(cmd)

    if result.returncode != 0:
        print(f"[失败] {input_video.name}")
        print(result.stderr)
        return False

    print(f"[完成] {input_video.name} -> {output_dir}")
    return True


def load_preview_frames(ffmpeg, input_video, sample_fps, width, height):
    """读取低分辨率灰度预览帧，用于判断 UI 是否真的变化。"""
    vf = f"fps={sample_fps},scale={width}:{height}:flags=area,format=gray"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(input_video),
        "-vf",
        vf,
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))

    frame_size = width * height
    data = result.stdout
    return [
        data[i : i + frame_size]
        for i in range(0, len(data) - frame_size + 1, frame_size)
    ]


def frame_diff(frame_a, frame_b):
    total = sum(abs(a - b) for a, b in zip(frame_a, frame_b))
    return total / (len(frame_a) * 255)


def select_stable_ui_frames(
    frames,
    sample_fps,
    stable_seconds,
    duplicate_threshold,
    stable_threshold,
):
    stable_step = max(1, round(sample_fps * stable_seconds))
    selected = []
    kept_frames = []

    for i in range(0, max(0, len(frames) - stable_step)):
        candidate = frames[i + stable_step]

        # 画面必须在一小段时间内保持稳定，才认为不是两个界面的交接/动画中间态。
        if frame_diff(frames[i], candidate) > stable_threshold:
            continue

        if any(frame_diff(kept, candidate) <= duplicate_threshold for kept in kept_frames):
            continue

        selected.append((i + stable_step) / sample_fps)
        kept_frames.append(candidate)

    return selected


def extract_png_at_time(ffmpeg, input_video, timestamp, output_file):
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(input_video),
        "-frames:v",
        "1",
        str(output_file),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def run_ui_extract(
    ffmpeg,
    input_video,
    output_dir,
    sample_fps,
    preview_width,
    preview_height,
    stable_seconds,
    duplicate_threshold,
    stable_threshold,
):
    clear_pngs(output_dir)

    print(f"[开始/UI筛选] {input_video.name}")
    frames = load_preview_frames(ffmpeg, input_video, sample_fps, preview_width, preview_height)
    if not frames:
        print(f"[失败] {input_video.name}: 没有读到视频帧")
        return False

    timestamps = select_stable_ui_frames(
        frames=frames,
        sample_fps=sample_fps,
        stable_seconds=stable_seconds,
        duplicate_threshold=duplicate_threshold,
        stable_threshold=stable_threshold,
    )

    if not timestamps:
        print(f"[失败] {input_video.name}: 没有筛选到稳定 UI 画面，可调低 --stable-seconds 或调高 --stable-threshold")
        return False

    for index, timestamp in enumerate(timestamps, start=1):
        extract_png_at_time(ffmpeg, input_video, timestamp, output_dir / f"{index:06d}.png")

    print(f"[完成/UI筛选] {input_video.name} -> {output_dir}，共 {len(timestamps)} 张")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="将视频导出为 PNG 图片。支持普通逐帧、ffmpeg 去重、UI 稳定画面筛选。"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="输入视频文件或文件夹路径，可传多个",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="frames_output",
        help="输出总目录，默认: frames_output",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="仅导出不同显示帧（使用 ffmpeg mpdecimate 去除连续重复/近重复帧）",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="适合 UI 录屏：只保留稳定且肉眼差异明显的界面，跳过切换/动画中间态",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=6,
        help="UI 模式每秒抽样多少帧用于判断，默认: 6",
    )
    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=0.35,
        help="UI 模式要求画面稳定多久才保留，默认: 0.35 秒",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.06,
        help="UI 模式去重阈值，越大越容易认为两张图相同，默认: 0.06",
    )
    parser.add_argument(
        "--stable-threshold",
        type=float,
        default=0.018,
        help="UI 模式稳定阈值，越小越严格过滤过渡帧，默认: 0.018",
    )
    parser.add_argument(
        "--preview-size",
        default="64x64",
        help="UI 模式用于相似度判断的小图尺寸，默认: 64x64",
    )

    args = parser.parse_args()

    if args.dedupe and args.ui:
        print("--dedupe 和 --ui 只能二选一。UI 录屏建议使用 --ui。")
        sys.exit(1)

    try:
        ffmpeg = find_ffmpeg()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    videos = collect_videos(args.inputs)
    if not videos:
        print("没有找到可处理的视频文件。")
        sys.exit(1)

    try:
        preview_width, preview_height = map(int, args.preview_size.lower().split("x", 1))
    except ValueError:
        print("--preview-size 格式错误，例如: 64x64")
        sys.exit(1)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for video in videos:
        subdir = output_root / video.stem
        if args.ui:
            ok = run_ui_extract(
                ffmpeg=ffmpeg,
                input_video=video,
                output_dir=subdir,
                sample_fps=args.sample_fps,
                preview_width=preview_width,
                preview_height=preview_height,
                stable_seconds=args.stable_seconds,
                duplicate_threshold=args.duplicate_threshold,
                stable_threshold=args.stable_threshold,
            )
        else:
            ok = run_ffmpeg(ffmpeg, video, subdir, dedupe=args.dedupe)

        if ok:
            success_count += 1

    print(f"\n处理完成：{success_count}/{len(videos)} 个视频成功。")


if __name__ == "__main__":
    main()
