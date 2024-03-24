import os

from training import training_unit

project_name = 'yuanshen'
# 第一步：生成配置文件
# training_unit.generate_config(project_name, 4)

# 第二步：预处理音频文件
training_unit.transcribe_audio_files(
    config_path='config.yml',
    project_name=project_name,
    in_dir=os.path.join('data',project_name, 'raws'),
    output_path=os.path.join('data', project_name, 'esd.list')
)

# 第三步：预处理标签文件
# training_unit.preprocess_data(
#     transcription_path="path/to/transcription",
#     cleaned_path=None,
#     train_path="path/to/train_set",
#     val_path="path/to/val_set",
#     config_path="path/to/config",
#     val_per_lang=100,
#     max_val_total=1000,
#     clean=True
# )

# 第四步：生成 BERT 特征文件

# 第五步：生成 CLAP 特征文件

