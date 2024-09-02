## EchoUDA

This is the code of EchoUDA (Unsupervised Domain Adaptation for Echocardiography Segmentation) based on Pytorch. 

We provide a small dataset in `./Echodata` to demo the codes.

### Installation

```shell
git clone https://github.com/qiang-Blazer/EchoUDA.git
```

### Prerequisites

```
albumentations==1.4.11
numpy==1.26.0
opencv-python==4.10.0.84
opencv-python-headless==4.9.0.80
openpyxl==3.1.2
packaging==23.2
pandas==2.1.2
scikit-image==0.22.0
scikit-learn==1.3.2
torch==1.13.1
torchaudio==0.13.1
torchmetrics==1.2.1
torchvision==0.14.1
tornado==6.4.1
tqdm==4.66.1
```

Install required packages:

```shell
pip install -r requirements.txt
```

### Pre-train

```shell
python main_seg_da_fc_pretrain.py --source_dataset EchoNet_Dynamic  --result_dir runs/0  --view 4CH --position LV
```

This will perform the pre-train stage of EchoUDA using source dataset. Results are in `runs/0`. You can choose the `source_dataset`, `view`, `position`

parameters:

| Argument Name  | Type  | Default Value       | Description                                                  |
| :------------- | :---- | :------------------ | :----------------------------------------------------------- |
| cuda_id        | str   | ‘0’                 | Cuda ID to use for computation.                              |
| epoch          | int   | 200                 | Number of epochs to train the model.                         |
| batch_size     | int   | 100                 | Mini-batch size for training.                                |
| pin_mem        | bool  | True                | Pin CPU memory in DataLoader for more efficient transfer to GPU, but requires more memory. |
| num_workers    | int   | 0                   | Number of worker threads to use in DataLoader.               |
| seed           | int   | 0                   | Random seed for reproducibility.                             |
| r_size         | int   | 112                 | Resized frame height and width.                              |
| lr             | float | 0.0001              | Starting learning rate for training.                         |
| scheduler      | str   | “cosineannealinglr” | Scheduler to use during training.                            |
| source_dataset | str   | “EchoNet_Dynamic”   | Source dataset to use, choices include ‘EchoNet_Dynamic’, ‘CardiacUDA’, ‘CAMUS’, ‘EchoNet_Pediatric’, ‘HMC_QU’. |
| view           | str   | “4CH”               | Echo view perspective.                                       |
| position       | str   | “LV”                | The part of the echo for segmentation.                       |
| result_dir     | str   | “runs/0”            | Directory to save the results.                               |
| pretrained_dir | str   | None                | Directory to load pretrained checkpoints from.               |

### Domain Adaptation

```shell
python main_seg_da_fc.py --source_dataset EchoNet_Dynamic --target_dataset CAMUS --pretrained_dir runs/0 --result_dir runs/seg_da_fc_ED2CM4LV/ --view 4CH --position LV
```

This will perform the domain adaptation stage of EchoUDA using source dataset and target dataset simultaneously. Pretrained results in `runs/0` are needed. You can choose the `source_dataset`,`target_dataset`, `view`, `position`

parameters:

| Argument Name  | Type  | Default Value       | Description                                                  |
| :------------- | :---- | :------------------ | :----------------------------------------------------------- |
| cuda_id        | str   | ‘0’                 | Cuda ID to use for computation.                              |
| epoch          | int   | 200                 | Number of epochs to train the model.                         |
| batch_size     | int   | 100                 | Mini-batch size for training.                                |
| pin_mem        | bool  | True                | Pin CPU memory in DataLoader for more efficient transfer to GPU, but requires more memory. |
| num_workers    | int   | 0                   | Number of worker threads to use in DataLoader.               |
| seed           | int   | 0                   | Random seed for reproducibility.                             |
| r_size         | int   | 112                 | Resized frame height and width.                              |
| lr             | float | 0.0001              | Starting learning rate for training.                         |
| scheduler      | str   | “cosineannealinglr” | Scheduler to use during training.                            |
| source_dataset | str   | “EchoNet_Dynamic”   | Source dataset to use for training.                          |
| target_dataset | str   | “CardiacUDA”        | Target dataset for domain adaptation.                        |
| view           | str   | “4CH”               | Echo view perspective.                                       |
| position       | str   | “LV”                | The part of the echo for segmentation.                       |
| result_dir     | str   | “runs/0”            | Directory to save the results.                               |
| pretrained_dir | str   | None                | Directory to load pretrained checkpoints from.               |
| lambda_da      | float | 1.0                 | Weight of the domain adaptation loss in the total loss.      |
