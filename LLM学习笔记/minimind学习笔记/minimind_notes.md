本教程参考官方 [readme](/Users/pengqianhan/Documents/GitHub/minimind-pq/README.md) 和 [不是Issue，一点个人训练minimind的记录 #26](https://github.com/jingyaogong/minimind/issues/26)

由于本人手头只有一台Macbook (M1 Pro)，因此这个项目只是用来debug和学习代码，完全没有训练出一个可用的模型。在大佬的代码基础上减少了epoch，同时在一个epoch内只用很少的数据进行训练，代码可正常运行从而可以学习代码运行的逻辑。以下是我学习代码的流程：

0. train tokenizer
   - 下载 | **【tokenizer训练集】** | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) / [百度网盘](https://pan.baidu.com/s/1yAw1LVTftuhQGAC1Y9RdYQ?pwd=6666) 文件为 tokenizer_train.jsonl 
   - [博客或视频讲解](https://www.bilibili.com/video/BV1KZ421M7di/?spm_id_from=333.880.my_history.page.click&vd_source=e587bac74600ca53ef886eea337fe87d)
   - 运行 train_tokenizer.py, 运行结束后，在’model/minimind_tokenizer/‘下得到merges.txt,tokenizer_config.json,tokenizer.json,vocab.json 四个文件
1. data_process.py 处理数据，为pretrain 数据集做准备
   - 下载 | **【Pretrain数据】**   | [Seq-Monkey官方](http://share.mobvoi.com:5000/sharing/O91blwPkY)  / [百度网盘](https://pan.baidu.com/s/1-Z8Q37lJD4tOKhyBs1D_6Q?pwd=6666) / [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) | 此处是从hugging face下载 mobvoi_seq_monkey_general_open_corpus.jsonl 文件,大小为 14.5GB,解压后为33.39GB
   - 运行 data_process.py ，处理mobvoi_seq_monkey_general_open_corpus.jsonl，
       - if process_type == 1:在dataset目录下生成了pretrain_data.bin和clean_seq_monkey.bin两个文件
       - process_type == 2:
       - process_type == 3:
2. 预训练model，[1-pretrain.py](1-pretrain.py)
   -  使用 ./dataset/pretrain_data.bin 来预训练，直接运行1-pretrain.py即可，运行结束后在./out 目录下保存一个pretrain_512.pth 的模型文件
3. 有监督微调（Supervised Fine-Tuning，SFT）[3-full_sft.py](3-full_sft.py)
   - 读取 './dataset/sft_data_single.csv' 文件来进行 full sft 训练，运行3-full_sft.py即可，结束后在./out 目录下保存一个 full_sft_512.pth 的模型文件
   - 读取 './dataset/sft_data_multi.csv' 文件来进行 full sft 训练，运行3-full_sft.py即可，结束后在./out 目录下保存一个 full_sft_512.pth 的模型文件，也可以修改‘ckp = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth’这行中文件名称
4. 现在可以运行2-eval.py 来进行评估
5. LoRA SFT,[4-lora_sft.py](4-lora_sft.py)
   - git clone https://huggingface.co/jingyaogong/minimind-v1-small
   - 在 https://huggingface.co/jingyaogong/minimind-v1-small/tree/main 下载 pytorch_model.bin， 然后放到 ./minimind_v1_small 目录下
   - 然后运行4-lora_sft.py即可,运行结束会在out文件夹下保存‘adapter_config.json’ 和 'adapter_model.safetensors'
   - note: 直接下载huggingface的模型，会报错，暂时没找到解决方法，因此下载模型到本地运行
   - [学习资料](https://zhuanlan.zhihu.com/p/672999750)
   - [peft 库](https://github.com/huggingface/peft)

6. 5-dpo_train.py
   - 在[hugging face](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main/dpo) 下载 dpo_dpo_zh_demo.json，然后放在'./dataset/dpo/'
   - 这部分代码还没跑通
