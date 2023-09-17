#!/bin/bash
cd /content

env TF_CPP_MIN_LOG_LEVEL=1

apt -y update -qq
wget https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O /content/libtcmalloc_minimal.so.4
env LD_PRELOAD=/content/libtcmalloc_minimal.so.4

apt -y install -qq aria2 libcairo2-dev pkg-config python3-dev
pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 -U
pip install -q xformers==0.0.20 triton==2.0.0 gradio_client==0.2.7 -U

#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
#sudo dpkg -i cuda-keyring_1.1-1_all.deb
#sudo apt-get update
#sudo apt-get -y install cuda
#wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
#sudo sh cuda_12.0.0_525.60.13_linux.run



git clone -b v2.6 https://github.com/camenduru/stable-diffusion-webui
git clone https://huggingface.co/embed/negative /content/stable-diffusion-webui/embeddings/negative
git clone https://huggingface.co/embed/lora /content/stable-diffusion-webui/models/Lora/positive
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d /content/stable-diffusion-webui/models/ESRGAN -o 4x-UltraSharp.pth
wget https://raw.githubusercontent.com/camenduru/stable-diffusion-webui-scripts/main/run_n_times.py -O /content/stable-diffusion-webui/scripts/run_n_times.py
git clone https://github.com/deforum-art/deforum-for-automatic1111-webui /content/stable-diffusion-webui/extensions/deforum-for-automatic1111-webui
git clone https://github.com/camenduru/stable-diffusion-webui-images-browser /content/stable-diffusion-webui/extensions/stable-diffusion-webui-images-browser
git clone https://github.com/camenduru/stable-diffusion-webui-huggingface /content/stable-diffusion-webui/extensions/stable-diffusion-webui-huggingface
git clone https://github.com/camenduru/sd-civitai-browser /content/stable-diffusion-webui/extensions/sd-civitai-browser
git clone https://github.com/kohya-ss/sd-webui-additional-networks /content/stable-diffusion-webui/extensions/sd-webui-additional-networks
git clone https://github.com/camenduru/sd-webui-tunnels /content/stable-diffusion-webui/extensions/sd-webui-tunnels
git clone https://github.com/etherealxx/batchlinks-webui /content/stable-diffusion-webui/extensions/batchlinks-webui
git clone https://github.com/camenduru/stable-diffusion-webui-catppuccin /content/stable-diffusion-webui/extensions/stable-diffusion-webui-catppuccin
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg /content/stable-diffusion-webui/extensions/stable-diffusion-webui-rembg
git clone https://github.com/ashen-sensored/stable-diffusion-webui-two-shot /content/stable-diffusion-webui/extensions/stable-diffusion-webui-two-shot
git clone https://github.com/thomasasfk/sd-webui-aspect-ratio-helper /content/stable-diffusion-webui/extensions/sd-webui-aspect-ratio-helper
git clone https://github.com/tjm35/asymmetric-tiling-sd-webui /content/stable-diffusion-webui/extensions/asymmetric-tiling-sd-webui
#aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt -d /content/stable-diffusion-webui/models/Stable-diffusion -o sd-v1-5-inpainting.ckpt
#aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/12873 -d /content/stable-diffusion-webui/models/Lora -o innievag.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/16677 -d /content/stable-diffusion-webui/models/Lora -o mix4.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/23250 -d /content/stable-diffusion-webui/models/Lora -o breastBetter.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/31284 -d /content/stable-diffusion-webui/models/Lora -o koreanDoll.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/7935 -d /content/stable-diffusion-webui/models/Stable-diffusion -o realisticVisionV51_v13-inpainting.safetensors
wget https://github.com/dhiimr/stable/raw/main/small_tits.pt -o /content/stable-diffusion-webui/embeddings/small_tits.pt 
wget https://github.com/dhiimr/stable/raw/main/breasts.pt -o /content/stable-diffusion-webui/embeddings/breasts.pt
cd /content/stable-diffusion-webui
git reset --hard
git -C /content/stable-diffusion-webui/repositories/stable-diffusion-stability-ai reset --hard
#aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.ckpt -d /content/stable-diffusion-webui/models/Stable-diffusion -o 512-inpainting-ema.ckpt
#aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/stable-diffusion-2-inpainting/raw/main/v2-inpainting-inference.yaml -d /content/stable-diffusion-webui/models/Stable-diffusion -o 512-inpainting-ema.yaml

sed -i -e '''/from modules import launch_utils/a\import os''' /content/stable-diffusion-webui/launch.py
sed -i -e '''/        prepare_environment()/a\        os.system\(f\"""sed -i -e ''\"s/dict()))/dict())).cuda()/g\"'' /content/stable-diffusion-webui/repositories/stable-diffusion-stability-ai/ldm/util.py""")''' /content/stable-diffusion-webui/launch.py
sed -i -e 's/\["sd_model_checkpoint"\]/\["sd_model_checkpoint","sd_vae","CLIP_stop_at_last_layers"\]/g' /content/stable-diffusion-webui/modules/shared.py
python launch.py --enable-insecure-extension-access --disable-safe-unpickle --xformers --no-hashing --disable-console-progressbars --ngrok=2UvgnHcqINMaPdZyLM0p6rgrcw4_3NdbUmE2pRd712jxVAdAJ --ngrok-region=jp --opt-sub-quad-attention --opt-channelslast --no-download-sd-model --gradio-queue --listen --ckpt-dir=/content/stable-diffusion-webui/models/Stable-diffusion --ckpt realisticVisionV51_v13-inpainting.safetensors --vae-dir=/content/stable-diffusion-webui/models/VAE --hypernetwork-dir=/content/stable-diffusion-webui/models/hypernetworks --embeddings-dir=/content/stable-diffusion-webui/embeddings --lora-dir=/content/stable-diffusion-webui/models/Lora --lowram --theme dark --no-half-vae
