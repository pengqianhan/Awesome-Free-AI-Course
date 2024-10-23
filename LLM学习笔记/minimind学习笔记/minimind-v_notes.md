## 

0. 在model/LMConfig.py中修改配置，dim=512和n_layers=8
1. 在 [hugging  face](https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/tree/main) 下载pretrain_images.zip 和 sft_images.zip 解压之后放在./dataset 目录下，然后下载LLaVA-Instruct 和 LLaVA-Pretrain 这两个文件夹以及里面的数据集，也放在 ./dataset 目录下
2. 点击链接[预训练权重](https://pan.baidu.com/s/1LE1SPoPYGS7VNtT1tpf7DA?pwd=6666)，下载512_llm.pth，放到./out/目录下
3. Mac 安装 git-lfs [教程地址](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
4. 将路径改为 model/clip_model 然后 下载clip-vit-base-patch32

   ```bash
    cd model/clip_model & git clone https://hf-mirror.com/openai/clip-vit-base-patch32
    ```
note: 下载完之后clip-vit-base-patch32下的文件和 [hugging face](https://huggingface.co/openai/clip-vit-base-patch32/tree/main) 内的文件是一样的

之后将路径改回项目根目录：cd minimind-v-mac 或者新开一个bash

5. 运行 1-pretrain_vlm.py 
  - 这一步主要训练 projector
  - 得到 out/512_vlm_pretrain.pth 
6. 运行