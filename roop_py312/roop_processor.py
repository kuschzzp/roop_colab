#!/usr/bin/env python3
"""
Standalone Roop Processor (兼容 Python 3.12)
包含所有核心处理逻辑的独立文件
"""

import os
import shutil
import sys
import threading
import warnings
from typing import Any, List

import cv2
import insightface
from insightface import model_zoo
# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Python 3.12: 兼容 collections.Iterable 移除问题
import collections
import collections.abc
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

# 过滤掉torchvision的弃用警告
warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", message=".*rcond parameter.*", category=FutureWarning)

# 导入必要的模块
import globals
from utilities import (
    has_image_extension, is_image, is_video, detect_fps, create_video,
    extract_frames, get_temp_frame_paths, restore_audio, create_temp,
    move_temp, clean_temp
)
from face_analyser import get_one_face, get_many_faces, find_similar_face
from face_reference import get_face_reference, set_face_reference, clear_face_reference

# 全局变量
FACE_SWAPPER = None
FACE_ENHANCER = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()


def get_face_swapper() -> Any:
    """获取面部交换器模型"""
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = os.path.join(os.path.dirname(__file__), 'models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=globals.execution_providers)
    return FACE_SWAPPER

# def get_face_swapper() -> Any:
#     global FACE_SWAPPER

#     with THREAD_LOCK:
#         if FACE_SWAPPER is None:
#             FACE_SWAPPER = model_zoo.get_model('inswapper_128')  # 使用模型名
#             FACE_SWAPPER.prepare(ctx_id=-1)  # -1 表示 CPU，如果有 GPU 可改 0
#     return FACE_SWAPPER


def clear_face_swapper() -> None:
    """清除面部交换器"""
    global FACE_SWAPPER
    FACE_SWAPPER = None


def get_device() -> str:
    """获取执行设备"""
    if 'CUDAExecutionProvider' in globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in globals.execution_providers:
        return 'mps'
    return 'cpu'


def get_face_enhancer() -> Any:
    """获取面部增强器"""
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            try:
                # ✅ Python 3.12 兼容导入（GFPGAN新版）
                from gfpgan import GFPGANer
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
        if globals.many_faces:
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
    reference_face = None if globals.many_faces else get_face_reference()

    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame, processors)
        cv2.imwrite(temp_frame_path, result)


def process_image(source_path: str, target_path: str, output_path: str, processors: List[str]) -> None:
    """处理单张图像"""
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if globals.many_faces else get_one_face(
        target_frame, globals.reference_face_position
    )
    result = process_frame(source_face, reference_face, target_frame, processors)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str], processors: List[str]) -> None:
    """处理视频"""
    if 'face_swapper' in processors and not globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, globals.reference_face_position)
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
    similar_face_distance: float = 0.85
) -> bool:
    """
    运行Roop处理器的主要函数
    """

    # 设置全局变量
    globals.source_path = source_path
    globals.target_path = target_path
    globals.output_path = output_path
    globals.frame_processors = frame_processors or ['face_swapper']
    globals.execution_providers = [execution_provider] if not execution_provider.endswith(
        'ExecutionProvider') else [execution_provider]
    globals.keep_fps = keep_fps
    globals.many_faces = many_faces
    globals.skip_audio = skip_audio
    globals.reference_face_position = reference_face_position
    globals.reference_frame_number = reference_frame_number
    globals.headless = True
    globals.temp_frame_format = 'png'
    globals.temp_frame_quality = 0
    globals.output_video_encoder = 'libx264'
    globals.output_video_quality = 35
    globals.similar_face_distance = similar_face_distance

    # 确保模型目录存在
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    try:
        # 处理图像
        if has_image_extension(target_path):
            shutil.copy2(target_path, output_path)
            process_image(source_path, output_path, output_path, globals.frame_processors)
            if is_image(output_path):
                update_status('Processing to image succeed!')
                return True
            else:
                update_status('Processing to image failed!')
                return False

        # 处理视频
        update_status('Creating temporary resources...')
        create_temp(target_path)

        if keep_fps:
            fps = detect_fps(target_path)
            update_status(f'Extracting frames with {fps} FPS...')
            extract_frames(target_path, fps)
        else:
            update_status('Extracting frames with 30 FPS...')
            extract_frames(target_path)

        temp_frame_paths = get_temp_frame_paths(target_path)
        if temp_frame_paths:
            update_status('Processing frames...')
            process_video(source_path, temp_frame_paths, globals.frame_processors)
        else:
            update_status('Frames not found...')
            return False

        if keep_fps:
            fps = detect_fps(target_path)
            update_status(f'Creating video with {fps} FPS...')
            create_video(target_path, fps)
        else:
            update_status('Creating video with 30 FPS...')
            create_video(target_path)

        if skip_audio:
            move_temp(target_path, output_path)
            update_status('Skipping audio...')
        else:
            if keep_fps:
                update_status('Restoring audio...')
            else:
                update_status('Restoring audio might cause issues as fps are not kept...')
            restore_audio(target_path, output_path)

        update_status('Cleaning temporary resources...')
        clean_temp(target_path)

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
        clear_face_swapper()
        clear_face_enhancer()
        clear_face_reference()


def main():
    success = run_processor(
        source_path='/workspaces/roop_colab/1.jpg',
        target_path='/workspaces/roop_colab/4.jpg',
        output_path='/workspaces/roop_colab/oooooooooooooooo.jpg',
        frame_processors=['face_swapper', 'face_enhancer'],
        execution_provider='CPUExecutionProvider',
        keep_fps=True,
        many_faces=False,
        similar_face_distance=0.80
    )

    if success:
        print("处理完成!")
    else:
        print("处理失败!")


if __name__ == '__main__':
    main()
