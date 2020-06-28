prepare:
	make -C input/birdsong-recognition


train:
	CUDA_VISIBLE_DEVICES=0 python train.py --config configs/000_ResNet50.yml
