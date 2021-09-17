# Codes for "TransVOS: Video Object Setmentation with Transformers"

This repository contains the official codes for [TransVOS: Video Object Setmentation with Transformers](https://arxiv.org/abs/2106.00588).
## Requirements
- torch >= 1.6.0
- torchvison >= 0.7.0
- ...

To installl requirements, run:
```bash
conda env update -n TransVOS --file requirements.yaml
```

## Data Organization
### Static images
We follow [AFB-URR](https://github.com/xmlyqing00/AFB-URR) to convert static images ([MSRA10K](https://mmcheng.net/msra10k/), [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [PASCAL-S](http://cbs.ic.gatech.edu/salobj/download/salObj.zip), [PASCAL VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/), [COCO](http://cocodataset.org/#download)) into a uniform format (followed DAVIS).
### Youtube-VOS
Download the [YouTube-VOS](https://youtube-vos.org/) dataset, then organize data as following format:
```
YTBVOS
      |----train
      |     |-----JPEGImages
      |     |-----Annotations
      |     |-----meta.json
      |----valid
      |     |-----JPEGImages
      |     |-----Annotations
      |     |-----meta.json 
```
Where `JPEGImages` and `Annotations` contain the frames and annotation masks of each video.

### DAVIS
Download the [DAVIS17](https://davischallenge.org/davis2017/code.html) datasets, then organize data as following format:
```
DAVIS
      |----JPEGImages
      |     |-----480p
      |----Annotations
      |     |-----480p (annotations for DAVIS 2017)
      |----ImageSets
      |     |-----2016
      |     |-----2017
      |----DAVIS-test-dev (data for DAVIS 2017 test-dev)
```

## Training
### Pretraining on static images
To pretrain the TransVOS network on static images, modify the dataset root (`$cfg.DATA.PRETRAIN_ROOT`) in `config.py`, then run following command.
```bash
python train.py --gpu ${GPU-IDS} --exp_name ${experiment} --pretrain
```
### Training on DAVIS17 & YouTube-VOS
To train the TransVOS network on DAVIS & YouTube-VOS, modify the dataset root (`$cfg.DATA.DAVIS_ROOT`, `$cfg.DATA.YTBVOS_ROOT`) in `config.py`, then run following command.
```bash
python train.py --gpu ${GPU-IDS} --exp_name ${experiment} --initial ${./checkpoints/*.pth.tar}
```
## Testing
Download the pretrained DAVIS17 [checkpoint](https://drive.google.com/file/d/1ebe_-ScD3UPQ3nquxNSb1pbmDEjQdOtG/view?usp=sharing) and YouTube-VOS [checkpoint](https://drive.google.com/file/d/1iDCGMLSUq6_wQDG3oA3M6Kqpds4U7JFN/view?usp=sharing).

To eval the TransVOS network on (DAVIS16/17), modify `$cfg.DATA.VAL.DATASET_NAME`, then run following command
```bash
python eval.py --checkpoint ${./checkpoints/*.pth.tar}
```
To test the TransVOS network on (DAVIS17 test-dev/youTube-vos), modify `$cfg.DATA.TEST.DATASET_NAME`, then run following command
```bash
python test.py --checkpoint ${./checkpoints/*.pth.tar}
```
The test results will be saved as indexed png file at `${results}/`.

Additionally, you can modify some setting parameters in `config.py` to change configuration.

# Acknowledgement
This codebase borrows the codes from [official AFB-URR repository](https://github.com/xmlyqing00/AFB-URR) and [official DETR repository](https://github.com/facebookresearch/detr).

# Citation
```
@article{mei2021transvos,
  title={TransVOS: Video Object Segmentation with Transformers},
  author={Mei, Jianbiao and Wang, Mengmeng and Lin, Yeneng and Liu, Yong},
  journal={arXiv preprint arXiv:2106.00588},
  year={2021}
}
