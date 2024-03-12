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