import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
from random import shuffle
from typing import Optional
import librosa
import soundfile
import yaml
from funasr import AutoModel
import torch
from multiprocessing import Pool
import commons
import utils
from tqdm import tqdm
from text import cleaned_text_to_sequence, get_bert, chinese
import argparse
import torch.multiprocessing as mp
from .slicer2 import Slicer
from clap_wrapper import get_clap_audio_feature
from config import config
import sys
from pypinyin import lazy_pinyin
from infer import latest_version
from concurrent.futures import ProcessPoolExecutor, as_completed


# from modelscope import pipeline, Tasks


# 第一步：生成配置文件
def get_path(data_dir):
    start_path = os.path.join("data", data_dir)
    lbl_path = os.path.join(start_path, "esd.list")
    train_path = os.path.join(start_path, "train.list")
    val_path = os.path.join(start_path, "val.list")
    config_path = os.path.join(start_path, "config.json")
    return start_path, lbl_path, train_path, val_path, config_path


def generate_config(data_dir, batch_size):
    assert data_dir != "", "数据集名称不能为空"
    start_path, _, train_path, val_path, config_path = get_path(data_dir)
    os.makedirs(start_path, exist_ok=True)
    if os.path.isfile(config_path):
        config = json.load(open(config_path, "r", encoding="utf-8"))
    else:
        config = json.load(open("configs/config.json", "r", encoding="utf-8"))

    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["train"]["batch_size"] = batch_size

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    if not os.path.exists("config.yml"):
        shutil.copy(src="default_config.yml", dst="config.yml")
    # 创建目录
    raw_path = os.path.join(start_path, "raws")
    wavs_path = os.path.join(start_path, "wavs")
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(wavs_path, exist_ok=True)
    # 复制底模
    shutil.copytree(os.path.join('filelists', 'models'), os.path.join(start_path, 'models'))

    return "配置文件生成完成"


# 第二步：预处理音频文件

# 音频切分 方案1
def split_audios(config_path, data_dir):
    """
    加载配置文件，获取模型名称，读取相应的音频文件，并切片保存。

    参数:
    config_path: 配置文件的路径。
    data_dir: 包含原始音频文件的数据目录，默认为'./data'。
    """
    # 从配置文件加载配置
    with open(config_path, mode="r", encoding="utf-8") as f:
        configyml = yaml.load(f, Loader=yaml.FullLoader)
    model_name = configyml["dataset_path"].replace("data/", "")

    # 读取音频文件
    for p in Path(data_dir).iterdir():
        for s in p.rglob('*.wav'):
            print('音频切片：', s)
            root = os.path.join(*s.parts[:-1])
            filename = s.name
            filename = filename[:filename.rindex('.')]  # 去后缀
            # 构建完整的音频文件路径
            audio_path = str(s)
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
            # 实例化Slicer类，根据配置切割音频文件
            slicer = Slicer(
                sr=sr,
                threshold=-40,
                min_length=2000,
                min_interval=300,
                hop_size=10,
                max_sil_kept=500
            )

            # 切片音频并保存切片
            chunks = slicer.slice(audio)
            # 确保输出目录存在
            output_dir = root

            # 保存切片后的音频文件
            for i, chunk in enumerate(chunks):
                # 如果音频是立体声，有多个通道，则转换为单一通道
                if chunk.ndim > 1:
                    chunk = chunk.T  # 交换轴来使音频文件为单通道（如果是立体声）
                # 仅保存时长超过1秒的切片
                if len(chunk) / sr > 1:
                    soundfile.write(os.path.join(output_dir, f'{filename}_{i}.wav'), chunk, sr)
                else:
                    print(f"音频片段 {filename}_{i}.wav 时长低于1秒，已丢弃。")
            # 删除原始音频文件
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"原始音频文件 {audio_path} 已被删除。")


def resample(in_dir, out_dir, sr=44100):
    """
    重采样音频文件。

    参数:
    in_dir: 包含音频文件的输入目录。
    out_dir: 输出目录。
    sr: 重采样后的采样率。
    """
    for audio_path in Path(in_dir).rglob('*.wav'):

        wav_path = os.path.join(audio_path)
        if os.path.exists(wav_path) and wav_path.lower().endswith(".wav"):
            print('重采样：', wav_path)
            wav, sr = librosa.load(wav_path, sr=sr)
            soundfile.write(wav_path, wav, sr)


def init_model():
    # 在这里初始化模型，这个函数将被调用来确保每个进程都有自己的模型实例
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
    # inference_pipeline = pipeline(
    #     task=Tasks.auto_speech_recognition,
    #     model='iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020', local_dir=local_dir_root,
    #     cache_dir=local_dir_root, model_revision="v2.0.4")
    # param_dict = {'use_timestamp': False}

    # 使用AutoModel推理，单进程需要5g显存左右
    model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                      vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                      punc_model="ct-punc-c", punc_model_revision="v2.0.4", local_dir=local_dir_root
                      # spk_model="cam++", spk_model_revision="v2.0.2",
                      )

    return model


def transcribe_audio_files(config_path, project_name, num_processes=2):
    """
    遍历指定文件夹中的 .wav 音频文件，重采样44100，进行语音识别，然后将转写结果写入文件。

    参数:
    config_path: 配置文件的路径。
    project_name: 使用的项目名称。
    in_dir: 包含音频文件的输入目录。
    output_path: 转写结果输出文件的路径。
    """
    in_dir = os.path.join('data', project_name, 'raws')
    output_path = os.path.join('data', project_name, 'esd.list')
    temp_path = os.path.join('data', 'temp')
    os.makedirs(temp_path, exist_ok=True)
    # 加载配置文件
    with open(config_path, mode="r", encoding="utf-8") as f:
        configyml = yaml.load(f, Loader=yaml.FullLoader)
    model_name = configyml["dataset_path"].replace("data/", "")
    # 如果in_dir文件夹有中文，修改文件夹名成拼音，并保存映射关系
    mapping_dict = {}
    for dir_name in os.listdir(in_dir):
        if os.path.isdir(os.path.join(in_dir, dir_name)):  # 确保是文件夹
            pinyin_name = ''.join(lazy_pinyin(dir_name))
            mapping_dict[pinyin_name] = dir_name
            # 如果中文名和拼音名相同，跳过
            if dir_name == pinyin_name:
                continue
            os.rename(os.path.join(in_dir, dir_name), os.path.join(in_dir, pinyin_name))
    # 音频重采样
    resample(in_dir, in_dir)
    print('音频重采样完成')
    # 切分音频
    split_audios(config_path, in_dir)
    print('切分音频完成')

    # 初始化模型
    model = init_model()

    lang2token = {
        'zh': "ZH|",
        'ja': "JP|",
        "en": "EN|",
    }
    # 获取总文件数并初始化处理计数器
    processed_files = 0

    # 初始化转写结果列表
    total_files = 0
    for wav_file in Path(in_dir).rglob('*.wav'):  # 使用.rglob()方法递归搜索
        total_files += 1  # 对每个找到的.wav文件增加计数

    speaker_annos = []
    print('音频文件数量：', total_files)
    # 迭代输入目录中的所有 wav 文件
    for audio_path in Path(in_dir).rglob('*.wav'):
        speaker_name = mapping_dict[audio_path.parent.name]
        try:
            # rec_result = inference_pipeline(input=str(audio_path), param_dict=param_dict)
            # lang, text = "zh", rec_result["text"]

            rec_result = model.generate(input=audio_path)
            lang, text = "zh", ''.join([i['text'] for i in rec_result])

            # 获取识别文本和语种
            if lang not in lang2token:
                print(f"{lang} 语言不支持，忽略")
                continue
            # 如果文本大于512，跳过
            if len(text) > 512:
                print(f"{audio_path}文本大于512，忽略文本")
                continue
            # 构造输出文本行
            text_line = f"{audio_path}|{speaker_name}|{lang2token[lang]}{text.strip()}\n"
            # 添加到转写结果列表
            speaker_annos.append(text_line)
            processed_files += 1
            print(f"已处理: {processed_files}/{total_files}")
        except Exception as e:
            print('error:', audio_path, e)

    # 如果发现转写文本，写入输出文件
    if speaker_annos:
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in speaker_annos:
                f.write(line)
    else:
        print("警告：未找到任何可转写的音频文件。")


# 第三步：预处理标签文件, 数据预处理函数，用于生成训练集和验证集
def preprocess_text(data_dir):
    assert data_dir != "", "数据集名称不能为空"
    start_path, lbl_path, train_path, val_path, config_path = get_path(data_dir)
    lines = open(lbl_path, "r", encoding="utf-8").readlines()
    with open(lbl_path, "w", encoding="utf-8") as f:
        for line in lines:
            path, spk, language, text = line.strip().split("|")
            path = path.replace("\\", "/")
            spk_path = path.split("/")[-2]
            # 放到角色文件夹
            os.makedirs(os.path.join(start_path, "wavs", spk_path), exist_ok=True)
            shutil.copy(path, os.path.join(start_path, "wavs", spk_path))
            path = os.path.join(start_path, "wavs", spk_path, os.path.basename(path))
            f.writelines(f"{path}|{spk}|{language}|{text}\n")
    preprocess_data(lbl_path, train_path, val_path, config_path)
    print("标签文件预处理完成")


def preprocess_data(
        transcription_path: str,
        train_path: str,
        val_path: str,
        config_path: str,
        val_per_lang=4,
        max_val_total=12,
        clean=True,
        cleaned_path='',
):
    def clean_text(text, language):
        # 返回清洗后的文本、音素、声调和词到音素的映射
        norm_text = chinese.text_normalize(text)
        phones, tones, word2ph = chinese.g2p(norm_text)
        return norm_text, phones, tones, word2ph

    if cleaned_path == "" or cleaned_path is None:
        cleaned_path = transcription_path + ".cleaned"

    if clean:
        with open(cleaned_path, "w", encoding="utf-8") as out_file:
            with open(transcription_path, "r", encoding="utf-8") as trans_file:
                lines = trans_file.readlines()
                for line in lines:
                    try:
                        utt, spk, language, text = line.strip().split("|")
                        norm_text, phones, tones, word2ph = clean_text(text, language)
                        out_file.write(
                            "{}|{}|{}|{}|{}|{}|{}\n".format(
                                utt,
                                spk,
                                language,
                                norm_text,
                                " ".join(phones),
                                " ".join([str(i) for i in tones]),
                                " ".join([str(i) for i in word2ph]),
                            )
                        )
                    except Exception as e:
                        print(line)
                        print(f"生成训练集和验证集时发生错误！, 详细信息:\n{e}")

    transcription_path = cleaned_path
    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(transcription_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            spk_utt_map[language].append(line)
            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list = []
    val_list = []

    # 原始训练集和测试集划分
    # for spk, utts in spk_utt_map.items():
    #     shuffle(utts)
    #     val_list += utts[:val_per_lang]
    #     train_list += utts[val_per_lang:]
    # shuffle(val_list)
    # if len(val_list) > max_val_total:
    #     train_list += val_list[max_val_total:]
    #     val_list = val_list[:max_val_total]

    # 训练集和测试集划分
    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        split_point = int(len(utts) * 0.9)  # 计算90%数据的切分点
        train_list += utts[:split_point]  # 前90%数据作为训练集
        val_list += utts[split_point:]  # 后10%数据作为测试集

    shuffle(train_list)  # 如果需要，可以洗牌训练集
    shuffle(val_list)  # 如果需要，可以洗牌测试集

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    json_config = json.load(open(config_path, encoding="utf-8"))
    json_config["data"]["spk2id"] = spk_id_map
    json_config["data"]["n_speakers"] = len(spk_id_map)
    # 新增写入：写入训练版本、数据集路径
    json_config["version"] = latest_version
    json_config["data"]["training_files"] = os.path.normpath(train_path).replace("\\", "/")
    json_config["data"]["validation_files"] = os.path.normpath(val_path).replace("\\", "/")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_config, f, indent=2, ensure_ascii=False)

    print("训练集和验证集生成完成！")


# 第四步：生成 BERT 特征文件
def bert_gen(data_dir, num_processes=2):
    start_path, _, train_path, val_path, config_path = get_path(data_dir)

    config = utils.get_hparams_from_file(config_path)
    training_files, validation_files, add_blank = (
        config.data.training_files,
        config.data.validation_files,
        config.data.add_blank
    )
    lines = []
    with open(training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    with open(validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    add_blanks = [add_blank] * len(lines)
    total = len(lines)
    count = 0
    if lines:
        # 多进程
        # with Pool(processes=num_processes) as pool:
        #     for _ in tqdm(pool.imap_unordered(process_line_bert, zip(lines, add_blanks)), total=len(lines)):
        #         pass
        for line in lines:
            try:
                process_line_bert(line, add_blank)
                process_line_clap(line)

            except Exception as e:
                print('生成bert特征时出错：', line.split('|')[0], e)
            count += 1
            if count % 100 == 0:
                print('bert特征文件已处理：%s/%s' % (count, total))
    print(f"BERT generation complete, {total} .bert.pt files created!")


def process_line_bert(line, add_blank):
    device = config.bert_gen_config.device
    if config.bert_gen_config.use_multi_device:
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        if torch.cuda.is_available():
            gpu_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    # 创建目录
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)

    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")

    bert = get_bert(text, word2ph, language_str, device)
    assert bert.shape[-1] == len(phone)
    torch.save(bert, bert_path)


def process_line_clap(line):
    device = config.emo_gen_config.device
    if config.emo_gen_config.use_multi_device:
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        if torch.cuda.is_available():
            gpu_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")

    clap_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".emo.pt")
    if os.path.isfile(clap_path):
        return

    audio = librosa.load(wav_path)[0]
    # audio = librosa.resample(audio, 44100, 48000)

    clap = get_clap_audio_feature(audio, device)
    torch.save(clap, clap_path)


# 第五步：训练
def fit_linux(data_dir):
    '''
    修改文件的\\到/，适配Linux训练
    '''
    start_path, _, train_path, val_path, config_path = get_path(data_dir)
    for root, ds, fs in os.walk(start_path):
        for f in fs:
            fullname = os.path.join(root, f)
            if fullname.endswith(".json") or fullname.endswith(".list"):
                # 修改文件的\\到/
                print('Modify file：', fullname)
                file = open(fullname, 'r', encoding='utf-8')
                file_content = file.read()
                file.close()
                file = open(fullname, 'w', encoding='utf-8')
                file.write(file_content.replace('\\', '/'))
                file.close()
