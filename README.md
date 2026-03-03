# Spine-DCAN

[Spine-DCAN A Dual-Branch Context-Aware Network for Automatic Lumbar Spine Segmentation in MRI]<br/>
[ShitengMa](https://github.com/Ma-S-T),

## Overview

Accurate segmentation of lumbar spine structures from magnetic resonance imaging (MRI) is a critical requirement for computer-aided diagnosis and quantitative assessment of spinal diseases. However, reliable automatic lumbar segmentation remains challenging due to complex anatomical morphology and significant inter-patient variability. To address these challenges, we propose a dual-branch context-aware network, termed Spine-DCAN, for automatic lumbar MRI segmentation.Spine-DCAN adopts a dual-branch encoder architecture in which a Dual-Branch Context Extraction (DBCE) module is incorporated at each stage to learn complementary feature representations. One branch employs residual convolutional encoding to preserve structural continuity, while the other integrates a Multi-scale Context Reweighting Module (MCRM) that selectively enhances spatial regions and channel features with rich information through multi-scale convolutional representations. Furthermore, a Hierarchical Feature Fusion Module (HFFM) is introduced between the encoder and decoder to integrate low-level spatial details with high-level semantic information, thereby mitigating information loss during feature reconstruction.The proposed method was evaluated on a manually annotated dataset consisting of 426 sagittal T2-weighted lumbar MRI images. Experimental results demonstrate that Spine-DCAN consistently outperforms several state-of-the-art segmentation methods, achieving an average Dice Similarity Coefficient (DSC) of 95.242% and an HD95 value of 2.870. These findings indicate that Spine-DCAN provides a robust and effective solution for automatic lumbar MRI segmentation.

## Environment Setup

Please set up an environment with python=3.11 and Install the necessary dependencies according to the following commands

```commandline
pip install -r requirments.txt
```

## Download Dataset

Please access [**Spine-DCAN Dataset**](https://github.com/Ma-S-T/Spine-DCAN-Dateset) and proceed to download the dataset:

## Prepare data

Please partition the data from the downloaded **"Spine-DCAN"** into **Train**, **Vali** and **Test** folder. 

```bash
├── dataset(old)(old)
│     ├──Train
│     │   ├──p[*]
│     │   │  ├──aug[*].pt
│     │   │  ├──***
│     │   ├──***
│     │ 
│     ├──Vali
│     │   ├──p[**]
│     │   │   ├──aug[*].pt
│     │   │   ├──***
│     │   ├──***
│     │
│     ├──Test
│     │   ├──p[***]
│     │   │   ├──aug[*].pt
│     │   │   ├──***
│     │   ├──***
```

## Pretrained SymTC models

Below is the download link for the pretrained SymTC model checkpoint:

[**SymTC Checkpoint**](https://drive.google.com/drive/folders/1NLWaRFqM1L-d8jpd7KOVP3M_nK-S03ve?usp=sharing)

## Train/Evaluation

### Training SymTC Models

Execute the command to initiate the training of the model.

`python train.py --net_name SymTC --num_classes 12 --max_epochs 500 --batch_size_train 3 --batch_size_eval 3 --base_lr 0.0001 --device cuda:0`

Adjust the values for _max_epochs_, _batch_size_train_, _batch_size_eval_, _device_, and any other relevant parameters as necessary.

The optimal model checkpoint **(best.pth)** will be stored in the **result/SymTC** directory.

### Model Evaluation

Execute the command for segmentation evaluation

`python evaluation.py --net_name SymTC --device cuda:0 --num_classes 12 --batch_size_eval 1 --save_fig False`

Ensure that the results are stored in the **"result"** folder. Enable sample image generation by setting the parameter **save_fig** to _True_.

### Model Robust Evaluation

Execute the command to assess the robustness of the model.

`python robust_evaluation.py --net_name SymTC --device cuda:0 --num_classes 12 --batch_size_eval 1`

The results will be logged in a file within the **robust** folder, and the generated sample images 
will be saved under the **result/robustness_evaluation** directory.

The shift directions are indicated as follows:
0 -> Up, 1 -> Down, 2 -> Left, 3 -> Right


## Reference

* [TransUnet](https://github.com/Beckschen/TransUNet)
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations

```
@misc{chen2024symtc,
      title={SymTC: A Symbiotic Transformer-CNN Net for Instance Segmentation of Lumbar Spine MRI}, 
      author={Jiasong Chen and Linchen Qian and Linhai Ma and Timur Urakov and Weiyong Gu and Liang Liang},
      year={2024},
      eprint={2401.09627},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
