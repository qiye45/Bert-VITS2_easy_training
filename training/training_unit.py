import os
import json
import shutil
from pathlib import Path
from funasr import AutoModel
import torchaudio
import argparse
import torch
import yaml
from config import config
import sys
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.hub.snapshot_download import snapshot_download

# 第一步：生成配置文件
def generate_config(data_dir, batch_size):
    def get_path(data_dir):
        start_path = os.path.join("data", data_dir)
        lbl_path = os.path.join(start_path, "esd.list")
        train_path = os.path.join(start_path, "train.list")
        val_path = os.path.join(start_path, "val.list")
        config_path = os.path.join(start_path, "config.json")
        return start_path, lbl_path, train_path, val_path, config_path

    assert data_dir != "", "数据集名称不能为空"
    start_path, _, train_path, val_path, config_path = get_path(data_dir)
    os.makedirs(start_path,exist_ok=True)
    if os.path.isfile(config_path):
        config = json.load(open(config_path, "r", encoding="utf-8"))
    else:
        config = json.load(open("configs/config.json", "r", encoding="utf-8"))

    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["train"]["batch_size"] = batch_size

    model_path = os.path.join(start_path, "models")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    if not os.path.exists("config.yml"):
        shutil.copy(src="default_config.yml", dst="config.yml")
    # 创建目录
    raw_path = os.path.join(start_path, "raws")
    wavs_path = os.path.join(start_path, "wavs")
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(wavs_path, exist_ok=True)

    return "配置文件生成完成"

# 第二步：预处理音频文件
def transcribe_audio_files(config_path, project_name, in_dir, output_path):
    """
    遍历指定文件夹中的 .wav 音频文件，进行语音识别，然后将转写结果写入文件。

    参数:
    config_path: 配置文件的路径。
    project_name: 使用的项目名称。
    in_dir: 包含音频文件的输入目录。
    output_path: 转写结果输出文件的路径。
    """

    # 加载配置文件
    with open(config_path, mode="r", encoding="utf-8") as f:
        configyml = yaml.load(f, Loader=yaml.FullLoader)
    model_name = configyml["dataset_path"].replace("Data/", "")

    # 根据模型名称构建模型目录路径
    local_dir_root = "./models_from_modelscope"
    # model_dir = snapshot_download('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    #                               cache_dir=local_dir_root)
    # inference_pipeline = pipeline(
    #     task=Tasks.auto_speech_recognition,
    #     model=model_dir,
    #     vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    #     punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    #     cache_dir=local_dir_root
    #     # lm_model='damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch',
    #     # lm_weight=0.15,
    #     # beam_size=10,
    # )
    model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                      vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                      punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                      cache_dir=local_dir_root
                      # spk_model="cam++", spk_model_revision="v2.0.2",
                      )

    # 推理参数配置
    param_dict = {'use_timestamp': False}
    lang2token = {
        'zh': "ZH|",
        'ja': "JP|",
        "en": "EN|",
    }

    # 获取总文件数并初始化处理计数器
    processed_files = 0

    # 初始化转写结果列表
    total_files=0
    for p in Path(in_dir).iterdir():
        for s in p.rglob('*.wav'):
            total_files+=1
    speaker_annos = []

    # 迭代输入目录中的所有 wav 文件
    for p in Path(in_dir).iterdir():
        for s in p.rglob('*.wav'):
            root = os.path.join(*s.parts[:-1])
            filename = s.name
            speaker_name = s.parts[-1]
            try:
                # 构建完整的音频文件路径
                audio_path = os.path.join(root, filename)
                # 进行语音识别
                # rec_result = inference_pipeline(audio_in=audio_path, param_dict=param_dict)
                rec_result=model.generate(input=audio_path)
                # 获取识别文本和语种
                lang, text = "zh", ''.join([i['text'] for i in rec_result])
                if lang not in lang2token:
                    print(f"{lang} 语言不支持，忽略")
                    continue
                # 构造输出文本行
                text_line = f"{audio_path}|{speaker_name}|{lang2token[lang]}{text.strip()}\n"
                # 添加到转写结果列表
                speaker_annos.append(text_line)
                processed_files += 1
                print(f"已处理: {processed_files}/{total_files}")
            except Exception as e:
                print(e)

    # 如果发现转写文本，写入输出文件
    if speaker_annos:
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in speaker_annos:
                f.write(line)
    else:
        print("警告：未找到任何可转写的音频文件。")


