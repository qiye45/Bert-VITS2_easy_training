from funasr import AutoModel

model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.4", local_dir=local_dir_root
                  # spk_model="cam++", spk_model_revision="v2.0.2",
                  )

def recognize_speech(args):
    audio_path, speaker_name = args
    global processed_files
    # 推理参数配置
    lang2token = {
        'zh': "ZH|",
        'ja': "JP|",
        "en": "EN|",
    }

    try:
        # 进行语音识别，复制到临时目录，防止中文路径问题
        rec_result = model.generate(input=audio_path)
        lang, text = "zh", ''.join([i['text'] for i in rec_result])

        # 获取识别文本和语种
        if lang not in lang2token:
            print(f"{lang} 语言不支持，忽略")
            return
            # 构造输出文本行
        text_line = f"{audio_path}|{speaker_name}|{lang2token[lang]}{text.strip()}\n"
        # 添加到转写结果列表
        processed_files += 1

        return text_line
    except Exception as e:
        print(f'处理文件 {audio_path} 时发生错误:', e)
        return None

if __name__ == '__main__':
    # 获取音频文件列表
    audio_files = []
    processed_files = 0
    for audio_path in Path(in_dir).rglob('*.wav'):
        speaker_name = mapping_dict[audio_path.parent.name]
        audio_files.append((audio_path, speaker_name))
    total_files = len(audio_files)
    print('音频文件数量：', total_files)
    speaker_annos = []
    # 使用进程池并发进行语言识别
    with ProcessPoolExecutor(max_workers=num_processes, initializer=worker_initializer) as executor:
        # 提交任务时，每个recognize_speech调用需要一个audio_file作为参数
        future_to_audio_file = {executor.submit(recognize_speech, audio_file): audio_file for audio_file in audio_files}
        # 收集结果
        for future in as_completed(future_to_audio_file):
            audio_file = future_to_audio_file[future]
            try:
                result = future.result()
                # 处理结果，例如将它们添加到列表中
                speaker_annos.append(result)
            except Exception as exc:
                print('%r generated an exception: %s' % (audio_file, exc))
    # 对结果进行处理
    # speaker_annos = [res for res in results if res is not None]

    # 统计处理完成的文件数
    processed_files = len(speaker_annos)
    print(f"已处理: {processed_files}/{total_files}")

    # 写入转写结果文件
    if speaker_annos:
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in speaker_annos:
                f.write(line)
    else:
        print("警告：未找到任何可转写的音频文件。")