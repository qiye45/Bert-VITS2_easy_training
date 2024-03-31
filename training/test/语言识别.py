from funasr import AutoModel
import torch

print(torch.__version__)
print(torch.cuda.is_available())
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
# 下载链接：https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary
local_dir_root = './models'
model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.4", local_dir=local_dir_root,
                  cache_dir=local_dir_root
                  # spk_model="cam++", spk_model_revision="v2.0.2",
                  )
res = model.generate(input=f"c1e82a9850d10f90.wav",
                     batch_size_s=300)
print(res)
