> [!NOTE]
> 🎉 We are happy to announce that our paper was accepted at EMNLP 2025! We are looking forward to present our poster in November in Suzhou. If you are attending the conference, come visit us! We will also talk about the dataset at the NLLP workshop in the afternoon.

# Evaluation Script for Headnote Generation on SLDS

This repository contains the script that can be used to evaluate LLMs on their ability to generate headnotes using the [SLDS dataset](https://huggingface.co/datasets/ipst/slds). Due to recent changes to `lighteval`, the task implementation has slightly changed, and we are currently in the process of integrating the task as an official community task. For now, this repository contains everything that you need to get started.

# Getting Started

## Installation

### Dependencies

To use this repository, you need to install uv. Please refer to their [documentation](https://docs.astral.sh/uv/getting-started/installation/) for further instructions. We recommend a system wide installation of uv over an installation via pip.

Once you have uv installed, all you need to do is running the following command:

```bash
uv sync
```

### API Key

To be able to use our script, you will need an [OpenRouter](https://openrouter.ai/) API key. Once you have your key, store it in the `.env_template` file as an environment variable and rename the file to `.env`.

OpenRouter is always needed because we internally call the judge model over their API. Additionally, it allows you to evaluate any supported model on OpenRouter in a one-shot setting. If you want to evaluate one of the fine-tuned models or another model that runs locally, this is entirely possible, but the OpenRouter API key is still needed for the judge model. You can change this behavior by modifying the `custom_task/slds.py` file (search for the string `openrouter/deepseek/deepseek-chat`).

### Running Your First Experiment

To test whether everything is working, you can run the evaluation script in debug mode. This will only execute one sample per chosen decision-headnote language subset. To do so, execute the following command:

```bash
uv run evaluate.py --debug --decision_language de --headnote-language de
```

This will by default use the `openrouter/openai/gpt-4o` model, but you can specify another one by adding the `--model` CLI parameter followed by a model name prefixed with `openrouter/`. By default, a one-shot setting is used.

### Evaluating Fine-Tuned Models

To evaluate on of our (or your own) fine-tuned models, you first need to deploy it somewhere. We used [vLLM](https://github.com/vllm-project/vllm) to serve our fine-tuned models locally. You can do the same by installing the `vLLM` pip package and then running something like this:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ipst/Qwen2.5-7B-Instruct-SLDS \
    --max-model-len 32000 \
    --enable-prefix-caching \
    --seed 2025 \
    --gpu-memory-utilization 0.7
```

Once your model is running, you can specify the corresponding model when running the evaluation script prefixed with `hosted_vllm`, for example:

```
uv run evaluate.py --no-one-shot --model hosted_vllm/ipst/Qwen2.5-7B-Instruct-SLDS
```

The `--no-one-shot` CLI parameter will deactivate the one-shot setting (resulting in a zero-shot evaluation) and then the script will run the locally hosted model throug vLLM. We found this to be faster than when using the vLLM entrypoint in lighteval. This might however change over time.

### Inspecting Outputs

Once you run the script, a directory `results` will be created by lighteval. It contains both the overall results (in form of a JSON file), as well as details with the individual metrics and generations in form of parquet files. We encourage you to inspect these files if you want to get a better idea of what the models are generating.

**Note:** The requests to the inference APIs go through litellm and they are cached. If you want to delete the cache, remove the folder `.litellm_cache` that is atuomatically generated upon the first request.

### A Note on Metrics

When you run the evaluation, you will see a table with metrics in the end. Please note that most metrics are multiplied with 100 for convenience. This is not the case with the ROUGE scores. Keep that in mind when evaluating the models.

# Troubleshooting

    I receive an error that states that pip is not installed.

In the above case, try to run the following command first: `uv pip install pip`. Afterwards the problem should hopefully disappear.

# References

If you use SLDS, please cite our preprint:

```bibtex
@article{rolshoven2024unlocking,
  title={Unlocking legal knowledge: A multilingual dataset for judicial summarization in Switzerland},
  author={Rolshoven, Luca and Rasiah, Vishvaksenan and Bose, Srinanda Br{\"u}gger and Hostettler, Sarah and Burkhalter, Lara and St{\"u}rmer, Matthias and Niklaus, Joel},
  journal={arXiv preprint arXiv:2410.13456},
  year={2024}
}
```