import os

from training import training_unit

project_name = 'live'
# 第一步：生成配置文件
# training_unit.generate_config(project_name, 2)

# 第二步：预处理音频文件
training_unit.transcribe_audio_files(
    config_path='config.yml',
    project_name=project_name
)

# 第三步：预处理标签文件
# training_unit.preprocess_text(project_name)

# 第四步：生成 BERT、clap 特征文件
# training_unit.bert_gen(project_name)

# 第五步：开始训练
# training_unit.fit_linux(project_name)
