# OVMS

## Prepare images

```
git lfs install
git clone https://huggingface.co/datasets/likaixin/ScreenSpot-Pro ./data/ScreenSpot-Pro
```

## Client

```
python3 -m venv venv
source venv/bin/activate
pip install openai torch torchvision "transformers<4.58" --extra-index-url "https://download.pytorch.org/whl/cpu"
```

```
python3 eval_screenspot_pro_parallel.py      --model_type "qwen3vl-ovms"      --screenspot_imgs "./data/ScreenSpot-Pro/images"      --screenspot_test "./data/ScreenSpot-Pro/annotations"      --task "all"     --language "en"     --gt_type "positive"     --log_path "./results/Qwen/Qwen3-VL-2B-Instruct.json"     --inst_style "instruction" --max_tasks 1
```

```
MODEL_NAME = os.environ.get("MODEL_NAME", "vlm")
BASE_URL = os.environ.get("OPENAI_API_BASE", "http://ov-ptl-13.sclab.intel.com:8181/v3")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "abc")
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", "99999999"))
MAX_TOKENS = int(os.environ.get("MAX_PIXELS", "10000"))
```

## Server

```
python export_model.py text_generation --source_model Qwen/Qwen3-VL-8B-Instruct --weight-format int4 --pipeline_type VLM_CB --model_name Qwen/Qwen3-VL-8B-Instruct --config_file_path models/config.json --model_repository_path vlm_models_with_export_models --overwrite_models --target_device GPU --tool_parser hermes3
```

```
docker run -it --rm --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -p 8181:8181 -v /home/devuser/dkalinow/vlm_models_with_export_models/Qwen/Qwen3-VL-2B-Instruct/:/model registry.toolbox.iotg.sclab.intel.com/openvino/model_server-gpu:dkalinow_ovms_ubuntu_qwen3-vl-gpu --rest_port 8181 --model_name vlm --model_path /model
```


Visualisation
```
python3 visualize_results.py --result results/Qwen/Qwen3-VL-2B-Instruct.json --output_dir visualized_results/Qwen3-VL-2B-Instruct
```

TODO:

- <|vision_start|><|image_pad|><|vision_end|> ?
- OpenVINO M-RoPE fixes


