#!/usr/bin/env python3
"""
Roop Simple Batch Processor
简化版批量处理脚本，路径直接在代码中配置
"""

import glob
import os
import shutil
import sys
import threading
import warnings
from typing import Any, List

import cv2
import insightface

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 过滤掉torchvision的弃用警告
warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", message=".*rcond parameter.*", category=FutureWarning)

# 导入必要的模块
import roop.globals
from roop.utilities import (
    has_image_extension, is_image, is_video, detect_fps, create_video,
    extract_frames, get_temp_frame_paths, restore_audio, create_temp,
    move_temp, clean_temp
)
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference

# 全局变量
FACE_SWAPPER = None
FACE_ENHANCER = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()

# ==================== 配置区域 ====================
# 请在这里修改您的路径配置
SOURCE_PATH = "/Users/kusch/Downloads/sharewithwindows/tmp/faces/xm2.jpg"  # 源人脸图片路径
TARGET_DIR = "/Users/kusch/Downloads/sharewithwindows/tmp/tobehandle"  # 目标媒体文件夹路径
OUTPUT_DIR = "/Users/kusch/Downloads/sharewithwindows/tmp/result"  # 输出文件夹路径

# 处理器配置
FRAME_PROCESSORS = ['face_swapper', 'face_enhancer']  # 可选: 'face_swapper', 'face_enhancer'
EXECUTION_PROVIDER = 'CPUExecutionProvider'  # 可选: 'CPUExecutionProvider', 'CoreMLExecutionProvider'
KEEP_FPS = True
MANY_FACES = False
SKIP_AUDIO = False


# ==================== 配置区域 ====================


def get_face_swapper() -> Any:
    """获取面部交换器模型"""
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = os.path.join(os.path.dirname(__file__), 'models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    """清除面部交换器"""
    global FACE_SWAPPER
    FACE_SWAPPER = None


def get_device() -> str:
    """获取执行设备"""
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'


def get_face_enhancer() -> Any:
    """获取面部增强器"""
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            try:
                from gfpgan.utils import GFPGANer
                model_path = os.path.join(os.path.dirname(__file__), 'models/GFPGANv1.4.pth')
                FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_device())
            except ImportError:
                print("警告: GFPGAN未安装，面部增强功能不可用")
                FACE_ENHANCER = None
    return FACE_ENHANCER


def clear_face_enhancer() -> None:
    """清除面部增强器"""
    global FACE_ENHANCER
    FACE_ENHANCER = None


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    """更新状态信息"""
    print(f'[{scope}] {message}')


def swap_face(source_face: Any, target_face: Any, temp_frame: Any) -> Any:
    """执行面部交换"""
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def enhance_face(target_face: Any, temp_frame: Any) -> Any:
    """执行面部增强"""
    face_enhancer = get_face_enhancer()
    if face_enhancer is None:
        return temp_frame

    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    padding_x = int((end_x - start_x) * 0.5)
    padding_y = int((end_y - start_y) * 0.5)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size:
        with THREAD_SEMAPHORE:
            _, _, temp_face = face_enhancer.enhance(temp_face, paste_back=True)
        temp_frame[start_y:end_y, start_x:end_x] = temp_face
    return temp_frame


def process_frame(source_face: Any, reference_face: Any, temp_frame: Any, processors: List[str]) -> Any:
    """处理单帧图像"""
    # 面部交换处理
    if 'face_swapper' in processors:
        if roop.globals.many_faces:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                for target_face in many_faces:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
        else:
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                temp_frame = swap_face(source_face, target_face, temp_frame)

    # 面部增强处理
    if 'face_enhancer' in processors:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = enhance_face(target_face, temp_frame)

    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], processors: List[str]) -> None:
    """处理多帧图像"""
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()

    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame, processors)
        cv2.imwrite(temp_frame_path, result)


def process_image(source_path: str, target_path: str, output_path: str, processors: List[str]) -> None:
    """处理单张图像"""
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame,
                                                                       roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame, processors)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str], processors: List[str]) -> None:
    """处理视频"""
    if 'face_swapper' in processors and not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)
    process_frames(source_path, temp_frame_paths, processors)


def run_processor(
        source_path: str,
        target_path: str,
        output_path: str,
        frame_processors: List[str] = None,
        execution_provider: str = 'cpu',
        keep_fps: bool = True,
        many_faces: bool = False,
        skip_audio: bool = False,
        reference_face_position: int = 0,
        reference_frame_number: int = 0,
        similar_face_distance: float=0.85
) -> bool:
    """
    运行Roop处理器的主要函数
    
    参数:
    - source_path: 源图像路径（包含要替换的人脸）
    - target_path: 目标图像/视频路径（将被替换人脸的媒体）
    - output_path: 输出路径
    - frame_processors: 帧处理器列表 ['face_swapper', 'face_enhancer']
    - execution_provider: 执行提供者 'cpu', 'cuda' 等
    - keep_fps: 是否保持原视频帧率
    - many_faces: 是否处理所有检测到的人脸
    - skip_audio: 是否跳过音频处理
    - reference_face_position: 参考人脸位置
    - reference_frame_number: 参考帧编号
    """

    # 设置全局变量
    roop.globals.source_path = source_path
    roop.globals.target_path = target_path
    roop.globals.output_path = output_path
    roop.globals.frame_processors = frame_processors or ['face_swapper']
    roop.globals.execution_providers = [execution_provider] if not execution_provider.endswith(
        'ExecutionProvider') else [execution_provider]
    roop.globals.keep_fps = keep_fps
    roop.globals.many_faces = many_faces
    roop.globals.skip_audio = skip_audio
    roop.globals.reference_face_position = reference_face_position
    roop.globals.reference_frame_number = reference_frame_number
    roop.globals.headless = True
    roop.globals.temp_frame_format = 'png'
    roop.globals.temp_frame_quality = 0
    roop.globals.output_video_encoder = 'libx264'
    roop.globals.output_video_quality = 35
    roop.globals.similar_face_distance = similar_face_distance

    # 确保模型目录存在
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    try:
        # 处理图像到图像
        if has_image_extension(target_path):
            shutil.copy2(target_path, output_path)
            # 处理帧
            process_image(source_path, output_path, output_path, roop.globals.frame_processors)
            # 验证图像
            if is_image(output_path):
                update_status('Processing to image succeed!')
                return True
            else:
                update_status('Processing to image failed!')
                return False

        # 处理视频
        update_status('Creating temporary resources...')
        create_temp(target_path)

        # 提取帧
        if keep_fps:
            fps = detect_fps(target_path)
            update_status(f'Extracting frames with {fps} FPS...')
            extract_frames(target_path, fps)
        else:
            update_status('Extracting frames with 30 FPS...')
            extract_frames(target_path)

        # 处理帧
        temp_frame_paths = get_temp_frame_paths(target_path)
        if temp_frame_paths:
            update_status('Processing frames...')
            process_video(source_path, temp_frame_paths, roop.globals.frame_processors)
        else:
            update_status('Frames not found...')
            return False

        # 创建视频
        if keep_fps:
            fps = detect_fps(target_path)
            update_status(f'Creating video with {fps} FPS...')
            create_video(target_path, fps)
        else:
            update_status('Creating video with 30 FPS...')
            create_video(target_path)

        # 处理音频
        if skip_audio:
            move_temp(target_path, output_path)
            update_status('Skipping audio...')
        else:
            if keep_fps:
                update_status('Restoring audio...')
            else:
                update_status('Restoring audio might cause issues as fps are not kept...')
            restore_audio(target_path, output_path)

        # 清理临时资源
        update_status('Cleaning temporary resources...')
        clean_temp(target_path)

        # 验证视频
        if is_video(output_path):
            update_status('Processing to video succeed!')
            return True
        else:
            update_status('Processing to video failed!')
            return False

    except Exception as e:
        update_status(f'Processing failed with error: {str(e)}')
        return False
    finally:
        # 清理资源
        clear_face_swapper()
        clear_face_enhancer()
        clear_face_reference()


def batch_process() -> bool:
    """
    批量处理目标目录中的所有图片和视频文件
    """

    # 检查源文件是否存在
    if not os.path.exists(SOURCE_PATH):
        update_status(f"错误: 源文件 {SOURCE_PATH} 不存在")
        return False

    # 检查目标目录是否存在
    if not os.path.exists(TARGET_DIR):
        update_status(f"错误: 目标目录 {TARGET_DIR} 不存在")
        return False

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 支持的文件扩展名
    image_extensions = ['*.jpg', '*.jpeg', '*.[pP][nN][gG]', '*.bmp', '*.tiff', '*.webp', '*.[jJ][pP][jG]']
    video_extensions = ['*.[mM][pP]4', '*.[aA][vV][iI]', '*.[mM][oO][vV]', '*.[mM][kK][vV]', '*.[fF][lL][vV]',
                        '*.[wW][mM][vV]']

    # 获取所有目标文件
    target_files = []

    # 收集所有图片文件
    for ext in image_extensions:
        target_files.extend(glob.glob(os.path.join(TARGET_DIR, ext)))

    # 收集所有视频文件
    for ext in video_extensions:
        target_files.extend(glob.glob(os.path.join(TARGET_DIR, ext)))

    if not target_files:
        update_status(f'No image or video files found in {TARGET_DIR}')
        return False

    update_status(f'Found {len(target_files)} files to process')

    # 处理每个文件
    successful_count = 0
    for i, target_file in enumerate(target_files):
        try:
            update_status(f'Processing file {i + 1}/{len(target_files)}: {os.path.basename(target_file)}')

            # 生成输出文件路径
            filename = os.path.basename(target_file)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_swapped{ext}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            # 处理单个文件
            success = run_processor(
                source_path=SOURCE_PATH,
                target_path=target_file,
                output_path=output_path,
                frame_processors=FRAME_PROCESSORS,
                execution_provider=EXECUTION_PROVIDER,
                keep_fps=KEEP_FPS,
                many_faces=MANY_FACES,
                skip_audio=SKIP_AUDIO
            )

            if success:
                successful_count += 1
                update_status(f'Successfully processed: {filename}')
            else:
                update_status(f'Failed to process: {filename}')

        except Exception as e:
            update_status(f'Error processing {target_file}: {str(e)}')

    update_status(f'Batch processing complete. {successful_count}/{len(target_files)} files processed successfully.')
    return successful_count > 0


def main():
    """主函数"""
    print("Roop Simple Batch Processor")
    print("=" * 30)
    print(f"Source Path: {SOURCE_PATH}")
    print(f"Target Directory: {TARGET_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Frame Processors: {', '.join(FRAME_PROCESSORS)}")
    print(f"Execution Provider: {EXECUTION_PROVIDER}")
    print("=" * 30)

    # 执行批量处理
    success = batch_process()

    if success:
        print("\n批量处理完成!")
    else:
        print("\n批量处理失败!")


if __name__ == '__main__':
    main()
