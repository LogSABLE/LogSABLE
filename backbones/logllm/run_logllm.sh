#!/bin/bash
# Trains and evaluates LogLLM on one dataset and writes val/test predictions.
# Usage: bash backbones/logllm/run_logllm.sh <Liberty|BGL|HDFS>
# MICRO_BATCH (default 4) lowers memory use; the effective batch size stays 16.
L=$(cd "$(dirname "$0")" && pwd); DS=$1
R=$L/runs/$DS; mkdir -p "$R"; cd "$R"
[ -f test_preds.npy ] && [ -f val_preds.npy ] && { echo "already done"; exit 0; }
cp "$L"/code/*.py .
sed -i -E "s|^dataset_name = .*|dataset_name = '$DS'|; s|^Bert_path = .*|Bert_path = '$L/models/bert-base-uncased'|; s|^Llama_path = .*|Llama_path = '$L/models/Meta-Llama-3-8B'|; s|^data_path = .*|data_path = '$L/data/$DS/DATA.csv'|; s|^ft_path = .*|ft_path = '$R/ft_model'|" train.py eval.py
sed -i "s|^micro_batch_size = .*|micro_batch_size = ${MICRO_BATCH:-4}|" train.py
sed -i "s|DATA.csv|train.csv|" train.py; sed -i "s|DATA.csv|test.csv|" eval.py
sed -i "s|^    gt = dataloader.dataset.get_label()|    np.save('test_preds.npy', preds)\n    gt = dataloader.dataset.get_label()|" eval.py
sed -i -E "s|^(    trainModel\(model, dataloader, gradient_accumulation_steps, n_epochs_2_1, lr_2_1\))|\1\n    model.save_ft_model('$R/ft_stage2_1')|; s|^(    trainModel\(model, dataloader, gradient_accumulation_steps, n_epochs_2_2, lr_2_2\))|\1\n    model.save_ft_model('$R/ft_stage2_2')|" train.py
sed "s|test.csv|val.csv|; s|test_preds.npy|val_preds.npy|" eval.py > eval_val.py
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
[ -f ft_model/projector.pt ] || "$L"/env/bin/python train.py || { echo TRAIN_FAILED; exit 1; }
[ -f test_preds.npy ] || "$L"/env/bin/python eval.py
[ -f val_preds.npy ] || "$L"/env/bin/python eval_val.py
