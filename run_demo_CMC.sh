MODEL_PATH=./model_mmdd_yyyy/default/plainvit_base1024_cocolvis_sax1/001/checkpoints/last_checkpoint.pth

python3 ./segnext/demo.py \
--checkpoint=${MODEL_PATH} \
--gpu 0