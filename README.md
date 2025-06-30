# 项目名称
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2504.15171) 

# How to run AV-CIL-FFIR?

Note: The Code is borrowed from Weiguo Pian (AC-CIL) ICCV 2023, the baseline model and code you can found in (https://github.com/weiguoPian/AV-CIL_ICCV2023/tree/main)

1. Using the code of `utils/gen_audio_features.py` and the `utils/gen_visual_features.py` to extract the audio and visual features, and save to .npy format. The audio pretrained model using the pretrained PANNs-MobileNetV2 (https://github.com/qiuqiangkong/audioset_tagging_cnn) and the visual pretrained model using the pretrained model S3D (https://github.com/kylemin/S3D).
2. You can run the script of extract_pretrained_feature.sh to save the features to /data
3. You need run the rim_AudioVisualBackone.sh to trained the AudioVisualBackbone first, because we are two stage model. The first stage we only trained the Red_Telapia fish.
4. Then you can use the pretrained model to train the HAIL-FFIA, you can directly use the run_incremental_hail.sh
6. The pre-trained model folder can be found at: https://drive.google.com/drive/folders/1fh-Lo3S7-aTgfPni5-IeG5_-P7MBKBfL?usp=drive_link


# AV-CIL-FFIR
Audio-Visual Class-Incremental Learning for Fish Feeding intensity Assessment in Aquaculture (The paper you can find on Arxiv: https://arxiv.org/abs/2504.15171)

The dataset will be released after the paper be accepted.


Abstract: Fish Feeding Intensity Recognition (FFIR) is crucial in industrial aquaculture management. Recent multi-modal approaches have shown promise in improving FFIR robustness and efficiency. However, these methods face significant challenges when adapting to new fish species or environments due to catastrophic forgetting, e.g., the loss of previously learned knowledge when adapting to new data. To address this limitation, we propose a novel hierarchical Audio-Visual Class Incremental Learning (HAIL) method for FFIR. More specifically, we develop a hierarchical representation learning method with a dual path knowledge preservation mechanism to disentangle general feeding intensity knowledge from fish-specific characteristics. In addition, we design a modality balancing scheme to adap tively adjust the importance of audio versus visual information based on the feeding stages. Unlike existing class incremental learning (CIL) methods, such as exemplar-based approaches, which require the storage of historical data, the proposed HAIL FFIR is a prototype-based approach that achieves exemplar free efficiency while preserving essential knowledge through compact feature representations, thus allowing subtle variations in feeding intensity to be distinguished for different fish species. To facilitate model development, we also created AV-CIL-FFIR, a new dataset comprising 81,932 labelled audio-visual clips capturing feeding intensities across six different fish species in real aquaculture environments. Experimental results show that HAIL-FFIR is superior to exemplar-based and exemplar-free methods on AV-CIL-FFIR, achieving higher accuracy with lower storage needs while effectively mitigating catastrophic forgetting in incremental learning for FFIR.


The AV-CIL-FFIR dataset was collected in a real aquaculture environment at the Aquatic Products Promotion Station in Guangzhou, China. We collected a total of 81,932 audio-visual samples, focusing on six different fish species commonly farmed in aquaculture: Tilapia, Black Perch, Jade Perch, Lotus Carp, Red Tilapia, and Sunfish. The real-world setting introduced various challenges such as environmental noise, water surface reflection, and foam, providing a more realistic and complex scenario compared to controlled environments.
 
 The dataset can be see bellow:

![image](https://github.com/FishMaster93/AV-CIL-FFIR/blob/main/images/CIL_dataset.png) 


The pipeline of the paper is shown below:

![image](https://github.com/FishMaster93/AV-CIL-FFIR/blob/main/images/framework.png) 