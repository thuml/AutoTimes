# AutoTimes

The repo is the official implementation for the paper: [AutoTimes: Autoregressive Time Series Forecasters via Large Language Models](https://arxiv.org/abs/2402.02370). It currently includes code implementations for the following tasks:

> **[Time Series Forecasting](./scripts/time_series_forecasting/)**: We repurpose large language models as out-of-box time series forecasters on benchmarks including long-term and short-term forecasting.

> **[Zero-shot Forecasting](./scripts/zero_shot_forecasting/)**: Large models exhibiting remarkable zero-shot capability are beneficial for data-scarce applications. AutoTimes takes advantage of this and demonstrates good performance without training samples.

> **[In-context Forecasting](./scripts/in_context_forecasting/)**: We propose in-context forecasting for the first time, where instructions in time series itself are available to further enhance forecasting.

> **[Generality on Large Language Models](scripts/method_generality)**: AutoTimes can be easily applied to various kinds of large language models, demonstrating generality and proper scaling behavior.

# Updates

:triangular_flag_on_post: **News** (2024.2) All the scripts for the above tasks in our [paper](https://arxiv.org/pdf/2402.02370.pdf) are available in this repo.


## Showcases

We provide several showcases of zero-shot and in-context forecasting results.

<p align="center">
<img src="./figures/showcases.png" alt="" align=center />
</p>


## Introduction

🌟 While prevalent forecasting models adopt the encoder-only structure with projection, we propose AutoTimes, a simple but effective way to convert generative LLMs as **autoregressive forecasters** with frozen parameters. **Token-wise** Prompting is also proposed to incorporate textual information (e.g. timestamps).

<p align="center">
<img src="./figures/motivation.png"  alt="" align=center />
</p>

💪 We aim to **fully revitalize the capabilities of LLMs as foundation models of time series**, including autoregressive token generation, zero-shot capability, in-context learning, and multimodal utilization.

🏆 AutoTimes demonstrate competitive results with existing baselines and have shown proficiency in **handling variable series lengths**: one model for all forecast lengths and gain with prolonged lookback.

## Overall Approach

* Establish the tokenization by the consistent training of Next Token Prediction.
* Ultilize inherent token transitions within the frozen LLM blocks to predict time series tokens 
* Prompted by textual covariates, such as timestamps aggregated in segments.

<p align="center">
<img src="./figures/method.png" alt="" align=center />
</p>

#### Comparsion with Other LLM4TS Methodology

<p align="center">
<img src="./figures/comparison.png"  alt="" align=center />
</p>

## Usage 

1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```

1. Put the datasets [[Google Drive]](https://drive.google.com/file/d/1t7jOkctNJ0rt3VMwZaqmxSuA75TFEo96/view?usp=sharing)
[[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/849427d3926f4fabbee7/) under the folder ```./dataset/```.

2. Download the large language models from [Hugging Face](https://huggingface.co/) and specify the model path using the `llm_ckp_dir` parameter in scripts.
   * [GPT2](https://huggingface.co/openai-community/gpt2)
   * [OPT Family](https://huggingface.co/facebook/opt-125m)
   * [LLaMA-7B](https://huggingface.co/meta-llama/Llama-2-7b)

3. Train and evaluate the model. We provide all the above tasks under the folder ```./scripts/```.

```
# the default large language model is LLaMA-7B

# long-term forecasting
bash ./scripts/time_series_forecasting/long_term/AutoTimes_ETTh1.sh

# short-term forecasting
bash ./scripts/time_series_forecasting/short_term/AutoTimes_M4.sh

# zero-shot forecasting
# it's worth noting that sM4_tM3 utilizes models trained
# on short-term, you should run AutoTimes_M4 first
bash ./scripts/zero_shot_forecasting/sM4_tM3.sh
bash ./scripts/zero_shot_forecasting/sM3_tM4.sh

# in-context forecasting
bash ./scripts/in_context_forecasting/M3.sh

# other large language models
bash ./scripts/method_generality/opt.sh

# preprocess timestamps to generate text embedding
python ./preprocess.py --gpu 0 --dataset ETTh1
```

## Zero-shot Forecasting

We evaluate the performance under the zero-shot scenario, where the forecaster is first trained on a source domain and then directly evaluated on the unseen target domain.

<p align="center">
<img src="./figures/zeroshot_results.png" alt="" align=center />
</p>

## In-context Forecasting

AutoTimes can **also utilize the instructions in time series**, where we propose **in-context forecasting**. Inspired by In-Context Learning of LLMs, we provide forecasting demonstration from the target domain as the prompt. The composed **time series sentence** is fed into our forecaster, which effectively empowers the prediction.

<p align="center">
<img src="./figures/in-context.png" alt="" align=center />
</p>

## Time Series Forecasting

AutoTimes demonstrates competitive performance in long-term and short-term scenarios. Notably, AutoTimes **adopts only one single model to tackle arbitrary forecast lengths by autoregression**, whereas other baselines necessitate training respectively on different lengths.

<p align="center">
<img src="./figures/short-term_results.png" alt="" align=center />
</p>

## Model Generality

We evaluate the efficiency of each repurposed LLM from three perspectives: forecasting performance, training speed, and parameters, demonstrating improved performance with the increase of parameters that **validates the scaling law**.

<p align="center">
<img src="./figures/llms.png" alt="" height = "300" align=center />
</p>

## Prompting Ablation

We conduct the ablation on Token-wise Prompting by integrating timestamps. The performance is **consistently promoted by the datetime information** across all datasets and forecasting lengths.

<p align="center">
<img src="./figures/ablation.png" alt="" align=center />
</p>

## Prolonged Lookbacks

As language models can generally give more accurate answers with a longer context, **the performance of AutoTimes is generally improving with the more available lookback observations**, which is highly desired in real-world applications.

## Parameter Efficiency

Despite LLM having a substantial amount of parameters, AutoTimes requires only minimal parameters (**up to 0.1%**) for training, acomplished by a single pair of MLPs for time series tokenization as the plugin of LLMs.

## Training on Custom Data

Tutorials are provided in this [repo](https://github.com/thuml/iTransformer/tree/main/scripts/multivariate_forecasting).


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
