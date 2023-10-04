#!/bin/bash
cd $HOME

env TF_CPP_MIN_LOG_LEVEL=1
conda install glib=2.51.0 -y
conda create -n glib-test -c defaults -c conda-forge python=3 glib=2.51.0 -y
pip install albumentations
pip install gdown
conda install -c bioconda aria2

wget https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O $HOME/libtcmalloc_minimal.so.4
env LD_PRELOAD=$HOME/libtcmalloc_minimal.so.4

#apt -y install -qq aria2 libcairo2-dev pkg-config python3-dev
pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 -U
pip install -q xformers==0.0.20 triton==2.0.0 gradio_client==0.2.7 -U




git clone -b v2.4 https://github.com/camenduru/stable-diffusion-webui
git clone https://huggingface.co/embed/negative $HOME/stable-diffusion-webui/embeddings/negative
git clone https://huggingface.co/embed/lora $HOME/stable-diffusion-webui/models/Lora/positive
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d $HOME/stable-diffusion-webui/models/ESRGAN -o 4x-UltraSharp.pth
wget https://raw.githubusercontent.com/camenduru/stable-diffusion-webui-scripts/main/run_n_times.py -O $HOME/stable-diffusion-webui/scripts/run_n_times.py
git clone https://github.com/deforum-art/deforum-for-automatic1111-webui $HOME/stable-diffusion-webui/extensions/deforum-for-automatic1111-webui
git clone https://github.com/camenduru/stable-diffusion-webui-images-browser $HOME/stable-diffusion-webui/extensions/stable-diffusion-webui-images-browser
git clone https://github.com/camenduru/stable-diffusion-webui-huggingface $HOME/stable-diffusion-webui/extensions/stable-diffusion-webui-huggingface
git clone https://github.com/camenduru/sd-civitai-browser $HOME/stable-diffusion-webui/extensions/sd-civitai-browser
git clone https://github.com/kohya-ss/sd-webui-additional-networks $HOME/stable-diffusion-webui/extensions/sd-webui-additional-networks
git clone https://github.com/camenduru/sd-webui-tunnels $HOME/stable-diffusion-webui/extensions/sd-webui-tunnels
git clone https://github.com/etherealxx/batchlinks-webui $HOME/stable-diffusion-webui/extensions/batchlinks-webui
git clone https://github.com/camenduru/stable-diffusion-webui-catppuccin $HOME/stable-diffusion-webui/extensions/stable-diffusion-webui-catppuccin
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg $HOME/stable-diffusion-webui/extensions/stable-diffusion-webui-rembg
git clone https://github.com/ashen-sensored/stable-diffusion-webui-two-shot $HOME/stable-diffusion-webui/extensions/stable-diffusion-webui-two-shot
git clone https://github.com/thomasasfk/sd-webui-aspect-ratio-helper $HOME/stable-diffusion-webui/extensions/sd-webui-aspect-ratio-helper
git clone https://github.com/tjm35/asymmetric-tiling-sd-webui $HOME/stable-diffusion-webui/extensions/asymmetric-tiling-sd-webui
gdown 1-H-yhExxn1SQ3Dbk62Nn4Hqr03HpkpAE -O $HOME/stable-diffusion-webui/models/Lora/
gdown 1-NpRmubKwqCvwWs3CBXM8mDFHp-0LNx2 -O $HOME/stable-diffusion-webui/models/Lora/
gdown 1-QqgpRPoZHEU3CoPu2jeSEEBoGdCLPSn -O $HOME/stable-diffusion-webui/models/Lora/

cd $HOME/stable-diffusion-webui
git reset --hard
git -C $HOME/stable-diffusion-webui/repositories/stable-diffusion-stability-ai reset --hard
gdown 107M-W8f11I4-sKwFqAKxxfS4Y412Vn3j -O $HOME/stable-diffusion-webui/models/Stable-diffusion/
sed -i -e '''/from modules import launch_utils/a\import os''' $HOME/stable-diffusion-webui/launch.py
sed -i -e '''/        prepare_environment()/a\        os.system\(f\"""sed -i -e ''\"s/dict()))/dict())).cuda()/g\"'' $HOME/stable-diffusion-webui/repositories/stable-diffusion-stability-ai/ldm/util.py""")''' $HOME/stable-diffusion-webui/launch.py
sed -i -e 's/\["sd_model_checkpoint"\]/\["sd_model_checkpoint","sd_vae","CLIP_stop_at_last_layers"\]/g' $HOME/stable-diffusion-webui/modules/shared.py
python launch.py --enable-insecure-extension-access --disable-safe-unpickle --xformers --no-hashing --disable-console-progressbars --ngrok=2UvgnHcqINMaPdZyLM0p6rgrcw4_3NdbUmE2pRd712jxVAdAJ --ngrok-region=jp --opt-sub-quad-attention --opt-channelslast --no-download-sd-model --gradio-queue --listen --ckpt-dir=$HOME/stable-diffusion-webui/models/Stable-diffusion --vae-dir=$HOME/stable-diffusion-webui/models/VAE --hypernetwork-dir=$HOME/stable-diffusion-webui/models/hypernetworks --embeddings-dir=$HOME/stable-diffusion-webui/embeddings --lora-dir=$HOME/stable-diffusion-webui/models/Lora --lowram --theme dark --no-half-vae --ckpt $HOME/stable-diffusion-webui/models/Stable-diffusion/realisticVisionV51_v13-inpainting.safetensors
