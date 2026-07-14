# ModPlant-LLM

Software repository for the modular-plant process-planning workflow described in **Verified LLM-Assisted Capability- and Skill-Based Process Planning Framework for Modular Plants** by Bowen Chen, Yuanchen Zhao, Torben Miny, and Tobias Kleinert.

`ModPlant-LLM` contains the software implementation associated with the paper, including the LoRA-finetuned language-model workflow, step-level FSA checking, the FSA+BFS+OPT verification/solver path, a Qt GUI, a notebook interface, packaged LoRA adapter directories, training/data-generation materials, and committed evaluation outputs.


## Paper Implementation

The paper studies how an ISA-88 General Recipe and a modular plant configuration can be turned into an executable process plan under capability, interface, capacity, and recipe-progress constraints. This repository contains the corresponding software implementation as two connected stages.

### Offline Training And Fine-Tuning

The offline stage produces solver-backed training material for the LLM:

- generate seed-based ModPlant configurations and orders,
- convert generated schedules into General Recipe XML and parsed JSON,
- construct reaction rules from the parsed recipe representation,
- use finite-state legality filtering, BFS reachability exploration, and optimization to produce executable reference operation sequences,
- convert solver-generated operation traces into next-step prediction samples,
- fine-tune LoRA adapters and evaluate the resulting models.

### Online Inference And Verification

The online stage runs the trained model inside the interactive software workflow:

- generate or load a ModPlant planning context,
- use the LoRA adapter for step-by-step next-operation prediction,
- check each newly generated LLM step with the FSA-style runtime checker,
- continue LLM generation only when the current step keeps the partial sequence feasible,
- automatically run FSA+BFS+OPT as a full recomputation path when the checker rejects an LLM step,
- optionally run FSA+BFS+OPT by user decision after a complete feasible LLM result.

The code is intended to make both stages of the paper workflow runnable and inspectable through the included notebooks, scripts, and GUI.

## What This Repository Contains

`ModPlant-LLM` implements the online neuro-symbolic planning workflow:

1. Generate a seed-based ModPlant configuration, order, recipe schedule, and reaction rules.
2. Use a LoRA adapter to run step-by-step LLM next-step prediction.
3. Check each generated step immediately against the FSA-style flow rules.
4. If the partial sequence remains feasible, continue with the next LLM step.
5. If any generated step fails the checker, stop the LLM path and run FSA+BFS+OPT to recompute a complete executable sequence.
6. If the LLM reaches a complete feasible result, allow the user to decide whether to also run FSA+BFS+OPT.

The LLM is therefore used as a fast candidate generator. The checker and symbolic solver remain part of the execution path, so this artifact does not claim that the LLM alone proves feasibility or optimality.

## Repository Layout

- [ModPlant-LLM_GUI.py](ModPlant-LLM_GUI.py)  
  Optional Qt GUI entry point for the interactive online workflow.

- [ModPlant-LLM.ipynb](ModPlant-LLM.ipynb)  
  Recommended notebook interface for running the online seed, LLM, FSA checker, and FSA+BFS+OPT workflow.

- [ModPlant_ui_lib](ModPlant_ui_lib)  
  Shared implementation package for the notebook/GUI interfaces, runtime session logic, LLM inference, FSA checker, recipe/rule helpers, and FSA+BFS+OPT integration.

- [Model](Model)  
  Packaged PEFT/LoRA adapter directories for the evaluated 1B, 3B, and 8B model variants. The default runtime adapter is `Model/3B-20260226_224018`.

- [Training](Training)  
  Fine-tuning dataset files and training-analysis notebook material.

- [Training-Data_Generation](Training-Data_Generation)  
  Offline data-generation notebooks and helper scripts for producing solver-backed operation-sequence corpora from ModPlant configurations, recipes, and reaction rules.

- [Evaluation](Evaluation)  
  Evaluation notebook, test data, and committed model-evaluation result folders.

## Requirements

The runtime needs the Python packages listed in [requirements.txt](requirements.txt), a working NVIDIA GPU runtime for LLM inference, and GLPK for the default open-source FSA+BFS+OPT solver path.

On Linux or WSL, the Qt GUI may also require the Qt/XCB/OpenGL runtime libraries listed in [requirements.txt](requirements.txt). If you only use [ModPlant-LLM.ipynb](ModPlant-LLM.ipynb) and do not launch [ModPlant-LLM_GUI.py](ModPlant-LLM_GUI.py), these GUI runtime libraries are usually not needed. The notebook interface and the Qt GUI expose the same core planning workflow and reuse the same backend session/pipeline logic.

Optional:

- `gurobipy` can be used for better solver performance.
- Gurobi requires you to obtain and configure your own license.

## Recommended GPU Docker Setup

The easiest way to run the notebook workflow is to start from an NVIDIA GPU environment and use the Unsloth Docker image. This avoids most manual CUDA, PyTorch, Transformers, and Unsloth installation work.

### 1. Prepare The Host

Make sure the machine has:

- an NVIDIA GPU,
- a working NVIDIA driver,
- Docker installed.

Windows users should first install WSL and Ubuntu 24.04 through WSL, then enable Docker's WSL integration in Docker Desktop. Native Linux users can skip the Windows/WSL steps.

### 2. Install NVIDIA Container Toolkit

Inside Ubuntu or your Linux shell, install the NVIDIA Container Toolkit:

```bash
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt-get update && sudo apt-get install -y \
  nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

If this is a fresh Docker installation, restart Docker after installing the toolkit.

### 3. Start The Unsloth Docker Container

From a working directory on the host, run:

```bash
docker run -d -e JUPYTER_PASSWORD="mypassword" \
  -p 8888:8888 -p 8000:8000 -p 2222:22 \
  -v $(pwd)/work:/workspace/work \
  --gpus all \
  unsloth/unsloth
```

Then either:

- open Jupyter at `http://localhost:8888`, using the password `mypassword`, or
- connect to the running container with VS Code and the Docker extension.

### 4. Add This Project

Download the release ZIP that includes the `Model/` adapter weights, extract it, and upload or copy the extracted `ModPlant-LLM` folder into the container.

Inside the container, install the remaining project-specific dependencies:

```bash
cd /ModPlant-LLM
python -m pip install -r requirements.txt
```

Also install GLPK in the container if it is not already available:

```bash
sudo apt-get update
sudo apt-get install -y glpk-utils
```

After that, use the notebook interface described below.

## Run The Notebook

The recommended interface is [ModPlant-LLM.ipynb](ModPlant-LLM.ipynb). After completing the Docker setup above, open Jupyter:

```bash
jupyter lab
```

Then open [ModPlant-LLM.ipynb](ModPlant-LLM.ipynb).

The notebook normally discovers the project root automatically from the current working directory. If Jupyter is launched from an unusual location and the setup cell cannot find the project, set `MODPLANT_LLM_ROOT` to the extracted `ModPlant-LLM` folder before running the notebook.

The Qt GUI entry point [ModPlant-LLM_GUI.py](ModPlant-LLM_GUI.py) exposes the same core workflow, but it is optional and may require additional GUI runtime libraries on Linux or WSL.

## Model And Release Notes

The repository is designed so the source tree can be published separately from large model-weight artifacts. A GitHub source upload may omit the bundled adapter weights, while a release ZIP can contain the ready-to-run `Model/` directories.

For the default experience, use the release ZIP containing `Model/3B-20260226_224018`, or set the adapter path in the GUI/notebook to another PEFT adapter directory that contains `adapter_config.json` and `adapter_model.safetensors`.

The default 3B adapter configuration records:

- base model: `unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit`
- max sequence length: `2560`
- LoRA rank: `32`
- effective batch size: `18`
- epochs: `4`

## Evaluation Snapshot

The committed evaluation report for `Model/3B-20260226_224018` records:

- base model: `unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit`
- total tested samples: `48`
- average score: `95.646701`
- average response time: `36.916257s`

These numbers come from [Evaluation/evaluation/TestResult-3B-20260226_224018](Evaluation/evaluation/TestResult-3B-20260226_224018). They document the committed test run in this repository; they should not be read as a general industrial benchmark.

## Expected Runtime Behavior

During an interactive run, the GUI or notebook shows:

- the seed-generated ModPlant configuration,
- the recipe schedule and reaction rules,
- LLM connect/process operation tables,
- FSA checker status and diagnostic text,
- optional or automatic FSA+BFS+OPT reference output,
- runtime logs and elapsed-time labels.

Temporary recipe and corpus artifacts are normally disabled unless artifact persistence is enabled in the interface.

## Scope and Limitations

- This repository is a research artifact for inspecting and demonstrating a neuro-symbolic planning workflow.
- It is not packaged as industrial deployment software.
- LLM output is validated step by step; invalid LLM output triggers the symbolic fallback path.
- Feasibility and fallback behavior depend on the FSA checker and FSA+BFS+OPT worker, not on unconstrained text generation alone.

## Citation

If you use this repository, please cite the associated paper:

> Bowen Chen, Yuanchen Zhao, Torben Miny, and Tobias Kleinert,  
> *Verified LLM-Assisted Process Planning for Modular Plants*.

See [CITATION.cff](CITATION.cff) for repository citation metadata.

## License & Acknowledgments

### Project License

The source code of `ModPlant-LLM` is released under the [MIT License](LICENSE).

### Third-Party Components

This project uses third-party libraries and tools distributed under their respective licenses:

- `PyQt-Fluent-Widgets`, used for the graphical user interface.
- `Unsloth`, used for efficient LLM loading, fine-tuning, and inference workflows.
- `GLPK`, used as the default open-source optimization solver through Pyomo.
- `Pyomo`, used to formulate and solve optimization models.
- `PyQt6`, `pandas`, `pyvis`, `JupyterLab`, and related Python packages listed in [requirements.txt](requirements.txt).

Users are responsible for reviewing and complying with the licenses of these third-party components, especially when redistributing Docker images, model weights, solver binaries, or packaged releases.
