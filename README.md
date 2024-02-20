# AutoTimes

The repo is the official implementation for the paper: [AutoTimes: Autoregressive Time Series Forecasters via Large Language Models](https://arxiv.org/abs/2402.02370). It currently includes code implementations for the following tasks:

> **[Multivariate Forecasting](./scripts/multivariate_forecasting/)**: We provide all scripts as well as datasets for the reproduction of forecasting results in this repo.

> **[Zero-shot Forecasting](./scripts/zeroshot_forecasting/)**:  iTransformer framework can consistently promote Transformer variants, and take advantage of the booming efficient attention mechanisms.

> **[In-context Forecasting](./scripts/in-context_forecasting/)**:  iTransformer framework can consistently promote Transformer variants, and take advantage of the booming efficient attention mechanisms.

> **[Generality on Large Language Models](scripts/llm_generality)**: iTransformer is demonstrated to generalize well on unseen time series, making it a nice alternative as the fundamental backbone of the large time series model.

# Updates

:triangular_flag_on_post: **News** (2024.3) All the scripts for the above tasks in our [paper](https://arxiv.org/pdf/2310.06625.pdf) are available in this repo.


## Introduction

🌟 Considering the characteristics of multivariate time series, iTransformer breaks the conventional model structure without the burden of modifying any Transformer modules. **Inverted Transformer is all you need in MTSF**.

<p align="center">
<img src="./figures/motivation.png"  alt="" align=center />
</p>

🏆 iTransformer achieves the comprehensive state-of-the-art in challenging multivariate forecasting tasks and solves several pain points of Transformer on extensive time series data.

<p align="center">
<img src="./figures/radar.png" height = "360" alt="" align=center />
</p>

😊 **iTransformer** is repurposed on the vanilla Transformer. We think the "passionate modification" of Transformer has got too much attention in the research area of time series. Hopefully, the mainstream work in the following can focus more on the dataset infrastructure and consider the scale-up ability of Transformer.



## Overall Architecture

iTransformer regards **independent time series as variate tokens** to **capture multivariate correlations by attention** and **utilize layernorm and feed-forward networks to learn series representations**.

<p align="center">
<img src="./figures/architecture.png" alt="" align=center />
</p>

The pseudo-code of iTransformer is as simple as the following:

<p align="center">
<img src="./figures/algorithm.png" alt="" align=center />
</p>

## Usage 

1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```

1. The datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/2ea5ca3d621e4e5ba36a/).

2. Train and evaluate the model. We provide all the above tasks under the folder ./scripts/. You can reproduce the results as the following examples:

```
# Multivariate forecasting with iTransformer
bash ./scripts/multivariate_forecasting/Traffic/iTransformer.sh

# Compare the performance of Transformer and iTransformer
bash ./scripts/boost_performance/Weather/iTransformer.sh

# Train the model with partial variates, and generalize on the unseen variates
bash ./scripts/variate_generalization/Electricity/iTransformer.sh

# Test the performance on the enlarged lookback window
bash ./scripts/increasing_lookback/Traffic/iTransformer.sh

# Utilize FlashAttention for acceleration
bash ./scripts/efficient_attentions/iFlashTransformer.sh
```

## Multivariate Forecasting
We evaluate the iTransformer on extensive challenging multivariate forecasting benchmarks as well as the server load prediction of Alipay online transactions (**generally hundreds of variates**, denoted as *Dim*). **Comprehensive good performance** (MSE/MAE) is achieved by iTransformer. iTransformer is particularly good at forecasting high-dimensional time series.

<p align="center">
<img src="./figures/datasets.png" alt="" align=center />
</p>

### Challenging Multivariate Time Series Forecasting Benchmarks (Avg Results)

<p align="center">
<img src="./figures/main_results.png" alt="" align=center />
</p>



### Online Transaction Load Prediction of Alipay Trading Platform (Avg Results) 

<p align="center">
<img src="./figures/main_results_alipay.png" alt="" align=center />
</p>

## Zero-shot Forecasting

By introducing the proposed framework, Transformer and its variants achieve **significant performance improvement**, demonstrating the **generality of the iTransformer approach** and **benefiting from efficient attention mechanisms**.

<p align="center">
<img src="./figures/boosting.png" alt="" align=center />
</p>

## In-context Forecasting

**Technically, iTransformer can forecast with arbitrary numbers of variables** during inference. We partition the variates of each dataset into five folders, train models with 20% variates, and use the partially trained model to forecast all varieties. iTransformers can be trained efficiently and forecast unseen variates with good generalizability.

<p align="center">
<img src="./figures/generability.png" alt="" align=center />
</p>

## Better Utilization of Lookback Windows

While previous Transformers do not necessarily benefit from the increase of historical observation. iTransformers show a surprising **improvement in forecasting performance with the increasing length of the lookback window**.

<p align="center">
<img src="./figures/increase_lookback.png" alt="" align=center />
</p>

## Model Analysis

Benefiting from inverted Transformer modules: 

- (Left) Inverted Transformers learn **better time series representations** (more similar [CKA](https://github.com/jayroxis/CKA-similarity)) favored by time series forecasting.
- (Right) The inverted self-attention module learns **interpretable multivariate correlations**.

<p align="center">
<img src="./figures/analysis.png" alt="" align=center />
</p>

- Visualization of the variates from Market and the learned multivariate correlations. Each variate represents the monitored interface values of an application, and the applications can be further grouped into refined categories.

<p align="center">
<img src="./figures/groups.png" alt="" align=center />
</p>

## Model Abalations

iTransformer that utilizes attention on variate dimensions and feed-forward on temporal dimension generally achieves the best performance. However, the performance of vanilla Transformer (the third row) performs the worst among these designs, **indicating the disaccord of responsibility when the conventional architecture is adopted**.

<p align="center">
<img src="./figures/ablations.png" alt="" align=center />
</p>

## Model Efficiency

We propose a training strategy for multivariate series by taking advantage of its variate generation ability. While the performance (Left) remains stable on partially trained variates of each batch with the sampled ratios, the memory footprint (Right) of the training process can be cut off significantly.

<p align="center">
<img src="./figures/efficient.png" alt="" align=center />
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
