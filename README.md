<h1 align="center">
  <img src="misc/art_fig.png" width="200" /></a><br>
  <b>Automated Design of Agentic Systems</b><br>
</h1>

<p align="center">
  <a href="https://github.com/ShengranHu/ADAS/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge"></a>
  <a href="https://arxiv.org/abs/2408.08435"><img src="https://img.shields.io/badge/arXiv-2408.08435-b31b1b.svg?logo=arxiv&style=for-the-badge"></a>
  <a href="https://www.shengranhu.com/ADAS/"><img src="https://img.shields.io/badge/-Website-%238D6748?style=for-the-badge&logo=Website&logoColor=white"></a>
  <a href="https://twitter.com/shengranhu/status/1825555341922480322"><img src="https://img.shields.io/badge/twitter-%230077B5.svg?&style=for-the-badge&logo=twitter&logoColor=white&color=00acee"></a>
</p>

<h3 align="center" style="display:inline-block; border:2px solid red; border-radius:4px; padding:4px; margin:4px 0;">
 <strong> ICLR 2025</strong><br>
</h3>
<h3 align="center" style="display:inline-block; border:2px solid red; border-radius:4px; padding:4px; margin:4px 0;">
  🏆 <strong>Outstanding Paper (NeurIPS 2024 Open-World Agent Workshop)</strong> 
  <a href="https://x.com/shengranhu/status/1868475359060226191" target="_blank" style="margin-left:8px;">[Tweet]</a>
</h3>

In this work, we describe a newly forming research area **A**utomated **D**esign of **A**gentic **S**ystems (**ADAS**), which aims to *automatically create powerful agentic system designs, including inventing novel building blocks and/or combining them in new ways.*


We present a simple yet effective ADAS algorithm named **Meta Agent Search** to demonstrate that **agents can invent novel and powerful agent designs**. In Meta Agent Search, a "meta" agent iteratively *programs* interesting new agents in code based on previous discoveries.


<p align="center">
<img src="misc/algo.png"/></a><br>
</p>

## Setup
```bash
conda create -n adas python=3.11
conda activate adas
pip install -r requirements.txt

# optional for OpenAI-compatible servers; a dummy value is fine for many local deployments
export OPENAI_API_KEY="EMPTY"

# optional; leave empty to use the provider default endpoint
export OPENAI_BASE_URL=""

# optional; defaults to GPT-OSS-20B in the patched scripts
export OPENAI_MODEL="GPT-OSS-20B"
```

## Running Instructions

### Running Meta Agent Search

To run experiments for each domain, navigate to its respective folder. The code in each folder is self-contained. Launch experiments using the `search.py` script located in each domain's folder.

```bash
python {DOMAIN}/search.py --model GPT-OSS-20B
```

Replace `{DOMAIN}` with the specific domain folder name {`_arc`, `_drop`, `_mgsm`, ...} to run the experiment for.

Across the benchmark scripts, `--n_repeat` is now the standardized CLI flag for repeated benchmark attempts, with `--n_repreat` kept as a backward-compatible alias. The main search scripts also support `--skip_evaluate` if you want to run only the search phase and skip the final evaluation pass.

The runtime now accepts any OpenAI-compatible chat-completions endpoint. If your open-source server does not support `response_format={"type": "json_object"}`, the code automatically retries without JSON mode and falls back to parsing the returned JSON text.

Each benchmark now writes per-query execution records to a local `runs/` folder inside that benchmark directory. Outputs are grouped under `runs/<expr_name>/attempt_<k>/` for search scripts, or an equivalent run-name derived from the evaluation file for transfer evaluation scripts. Within each attempt directory, a new CSV file is created for each evaluated solution or archive entry rather than combining an entire run into a single CSV. The filename includes the archive name, a timestamp, and the aggregate benchmark result for that evaluated solution. Each CSV contains the evaluated task, the generated code being executed, the output returned by that code, the expected output, the per-query score, and the serialized execution logs for all LLM invocations made by the agent while solving that query.

### Customizing Meta Agent Search for New Domains

You can easily adapt the code to search for new domains. To do so, follow these steps:

1. Modify the `evaluate_forward_fn()` function and adjust any necessary formatting prompts (e.g. [this line](https://github.com/ShengranHu/ADAS/blob/main/_mmlu/search.py#L89)) in the `search.py` file. 

2. Consider adding additional basic functions for the meta agent to utilize during the design process (similar to [this line](https://github.com/ShengranHu/ADAS/blob/main/_arc/search.py#L161)).

3. Update the domain-specific information within the prompts to match the requirements of your new domain (e.g. [this line](https://github.com/ShengranHu/ADAS/blob/main/_mmlu/mmlu_prompt.py#L229)).

4. Run the search and evaluation on your new domain.

### Safety Consideration
> [!WARNING]  
> The code in this repository involves executing untrusted model-generated code. We strongly advise users to be aware of this safety concern. While it is highly unlikely that model-generated code will perform overtly malicious actions in our current settings and with the models we use, such code may still act destructively due to limitations in model capability or alignment. By using this repository, you acknowledge and accept these risks.


## Citing
If you find this project useful, please consider citing:
```
@article{hu2024ADAS,
title={Automated Design of Agentic Systems},
author={Hu, Shengran and Lu, Cong and Clune, Jeff},
journal={arXiv preprint arXiv:2408.08435},
year={2024}
}
```
