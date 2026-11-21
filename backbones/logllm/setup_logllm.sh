#!/bin/bash
# Sets up LogLLM: official code (commit d1105daf), its Python environment, BERT and Llama-3-8B weights, and the input data.
set -e
L=$(cd "$(dirname "$0")" && pwd); cd "$L"
[ -d code ] || git clone https://github.com/guanwei49/LogLLM.git code
git -C code checkout d1105daf
rm -rf env
conda create -y -q -p ./env python=3.8
./env/bin/pip -q install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
./env/bin/pip -q install transformers==4.46.3 peft==0.13.2 accelerate==1.0.1 bitsandbytes==0.45.3 safetensors==0.5.3 tokenizers==0.20.3 datasets==3.1.0 scikit-learn==1.3.2 pandas==2.0.3 numpy==1.24.4 tqdm huggingface_hub==0.29.1
./env/bin/python -c "
import torch,transformers,peft,bitsandbytes
print('env ok',torch.__version__,'gpu',torch.cuda.is_available(),torch.cuda.get_device_name(0),transformers.__version__,peft.__version__,bitsandbytes.__version__)
from huggingface_hub import snapshot_download
print(snapshot_download('google-bert/bert-base-uncased',local_dir='$L/models/bert-base-uncased',allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt']))"
./env/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('NousResearch/Meta-Llama-3-8B',local_dir='$L/models/Meta-Llama-3-8B',allow_patterns=['LICENSE','USE_POLICY.md','config.json','generation_config.json','model*.safetensors','model.safetensors.index.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json']))"
sha256sum -c <<SHA
f2c144103072514542e327fa8080bd375cb300f2d453fba9ca3aea81d0d4cf33  models/Meta-Llama-3-8B/model-00001-of-00004.safetensors
d9eee5f23d94405d90b7e9ff88b9443fee42f8528a658f54214c2aba7530d80c  models/Meta-Llama-3-8B/model-00002-of-00004.safetensors
4b8fbc5e113f69768dd8de84661ea20af8a32b734a9976144b4236c447b40ccc  models/Meta-Llama-3-8B/model-00003-of-00004.safetensors
5dc34e6bdf2da9e35f0d93b5c333c870f3677dc43dc3a91ea3a8ad28a1fe1acb  models/Meta-Llama-3-8B/model-00004-of-00004.safetensors
SHA
PYTHONDONTWRITEBYTECODE=1 ${PYTHON:-python} "$L/prep_logllm.py"
echo SETUP_DONE
