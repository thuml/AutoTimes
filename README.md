# AutoTimes

The repo is the official implementation for the paper: [AutoTimes: Autoregressive Time Series Forecasters via Large Language Models](https://arxiv.org/abs/2402.02370). It currently includes code implementations for the following tasks:

> **[Time Series Forecasting](./scripts/multivariate_forecasting/)**: We provide all scripts as well as datasets for the reproduction of forecasting results in this repo.

> **[Zero-shot Forecasting](./scripts/zeroshot_forecasting/)**:  AutoTimes framework can consistently promote Transformer variants, and take advantage of the booming efficient attention mechanisms.

> **[In-context Forecasting](./scripts/in-context_forecasting/)**:  AutoTimes framework can consistently promote Transformer variants, and take advantage of the booming efficient attention mechanisms.

> **[Generality on Large Language Models](scripts/llm_generality)**: AutoTimes is demonstrated to generalize well on unseen time series, making it a nice alternative as the fundamental backbone of the large time series model.

# Updates

:triangular_flag_on_post: **News** (2024.3) All the scripts for the above tasks in our [paper](https://arxiv.org/pdf/2310.06625.pdf) are available in this repo.


## Introduction

🌟 Considering the characteristics of multivariate time series, AutoTimes breaks the conventional model structure without the burden of modifying any Transformer modules. **Inverted Transformer is all you need in MTSF**.

<p align="center">
<img src="./figures/motivation.png"  alt="" align=center />
</p>

🏆 AutoTimes achieves comprehensive state-of-the-art in challenging multivariate forecasting tasks and solves several pain points of Transformer on extensive time series data.

<p align="center">
<img src="./figures/comparison.png"  alt="" align=center />
</p>
😊 **AutoTimes** is repurposed on the vanilla Transformer. We think the "passionate modification" of Transformer has got too much attention in the research area of time series. Hopefully, the mainstream work in the following can focus more on the dataset infrastructure and consider the scale-up ability of Transformer.

## Overall Architecture

AutoTimes regards **independent time series as variate tokens** to **capture multivariate correlations by attention** and **utilize layernorm and feed-forward networks to learn series representations**.

<p align="center">
<img src="./figures/method.png" alt="" align=center />
</p>

## Usage 

1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```

1. The datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/2ea5ca3d621e4e5ba36a/).

2. Train and evaluate the model. We provide all the above tasks under the folder ./scripts/. You can reproduce the results as the following examples:

```
# Multivariate forecasting with AutoTimes
bash ./scripts/multivariate_forecasting/Traffic/AutoTimes.sh

# Compare the performance of Transformer and AutoTimes
bash ./scripts/boost_performance/Weather/AutoTimes.sh

# Train the model with partial variates, and generalize on the unseen variates
bash ./scripts/variate_generalization/Electricity/AutoTimes.sh

# Test the performance on the enlarged lookback window
bash ./scripts/increasing_lookback/Traffic/AutoTimes.sh

# Utilize FlashAttention for acceleration
bash ./scripts/efficient_attentions/iFlashTransformer.sh
```

## In-context Forecasting

**Technically, AutoTimes can forecast with arbitrary numbers of variables** during inference. We partition the variates of each dataset into five folders, train models with 20% variates, and use the partially trained model to forecast all varieties. AutoTimes can be trained efficiently and forecast unseen variates with good generalizability.

<p align="center">
<img src="./figures/in-context.png" alt="" align=center />
</p>
## Zero-shot Forecasting

By introducing the proposed framework, Transformer and its variants achieve **significant performance improvement**, demonstrating the **generality of the AutoTimes approach** and **benefiting from efficient attention mechanisms**.

<p align="center">
<img src="./figures/zeroshot_results.png" alt="" align=center />
</p>

## Long-term Forecasting

<p align="center">
<img src="./figures/long-term_results.png" alt="" align=center />
</p>

## Short-term Forecasting

<p align="center">
<img src="./figures/short-term_results.png" alt="" align=center />
</p>

## Model Generality

We propose a training strategy for multivariate series by taking advantage of its variate generation ability. While the performance (Left) remains stable on partially trained variates of each batch with the sampled ratios, the memory footprint (Right) of the training process can be cut off significantly.

<p align="center">
<img src="./figures/llms.png" alt="" height = "390" align=center />
</p>

## Prolonged Lookbacks

While previous Transformers do not necessarily benefit from the increase of historical observation. AutoTimes show a surprising **improvement in forecasting performance with the increasing length of the lookback window**.

<p align="center">
<img src="./figures/lookback.png" alt="" height = "350" align=center />
</p>

## Prompting Ablation

<p align="center">
<img src="./figures/ablation.png" alt="" align=center />
</p>

## Citation

If you find this repo helpful, please cite our paper. 

```
@article{liu2024autotimes,
  title={AutoTimes: Autoregressive Time Series Forecasters via Large Language Models},
  author={Liu, Yong and Qin, Guo and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2402.02370},
  year={2024}
}
```

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- FPT (https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)

## Contact

If you have any questions or want to use the code, feel free to contact:
* Yong Liu (liuyong21@mails.tsinghua.edu.cn)
* Guo Qin (qing20@mails.tsinghua.edu.cn)
