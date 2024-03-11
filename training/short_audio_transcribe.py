import os
import json
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


# 下载并设置语音模型的存储路径
local_dir_root = "./models_from_modelscope"
model_dir = snapshot_download('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
                              cache_dir=local_dir_root)

# 初始化自动语音识别任务的推理管道
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model=model_dir,
    vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    # 以下为未使用的模型配置，可以根据需要取消注释
    # lm_model='damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch',
    # lm_weight=0.15,
    # beam_size=10,
)

# 推理参数配置
param_dict = {'use_timestamp': False}
# 假设是支持的文件格式
supported_extensions = ['wav']

# 读取配置文件
with open('config.yml', mode="r", encoding="utf-8") as f:
    configyml = yaml.load(f, Loader=yaml.FullLoader)

model_name = configyml["dataset_path"].replace("Data/", "")

# 语言标识映射
lang2token = {
    'zh': "ZH|",
    'ja': "JP|",
    "en": "EN|",
}

# 进行单个音频文件的转写
def transcribe_one(audio_path):
    """进行单个音频文件的转写，并返回识别结果"""

    # 使用推理管道进行语音识别
    rec_result = inference_pipeline(audio_in=audio_path, param_dict=param_dict)
    # 打印并返回语言代码和识别文本
    print(rec_result["text"])
    return "zh", rec_result["text"]


if __name__ == "__main__":
    # 从配置对象中读取输入目录，并移除子目录部分
    parent_dir = config.resample_config.in_dir
    parent_dir = parent_dir.replace("\\audios", "")
    print('父级目录:', parent_dir)
    speaker = model_name

    # 初始化转写注释列表
    speaker_annos = []
    # 统计总文件数
    total_files = sum([len(files) for r, d, files in os.walk(parent_dir)])
    # 获取目标采样率
    with open(config.train_ms_config.config_path, 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']
    processed_files = 0  # 已处理文件数

    # 遍历所有文件
    for i, wavfile in enumerate(list(os.walk(parent_dir))[0][2]):
        try:
            # 进行语音文件转写
            lang, text = transcribe_one(os.path.join("./Data", speaker, "raw", wavfile))
            if lang not in lang2token:
                print(f"{lang} 语言不支持，忽略")
                continue
            # 构造输出文本格式
            text = f"./Data/{model_name}/wavs/{wavfile}|{model_name}|{lang2token[lang]}{text.strip()}\n"
            speaker_annos.append(text)
            processed_files += 1
            print(f"已处理: {processed_files}/{total_files}")
        except Exception as e:
            print(e)
            continue

    # 确认是否有转写注释产生
    if not speaker_annos:
        print("警告：未找到短音频文件。如果你只上传了长音频、视频或视频链接，那么这是预期内的。")
        print(
            "如果你上传了短音频的zip文件，而结果却没有短音频，那么这是不预期的。请检查你的文件结构或确认音频语言是否受支持。")

    # 将转写注释写入输出文件
    with open(config.preprocess_text_config.transcription_path, 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)