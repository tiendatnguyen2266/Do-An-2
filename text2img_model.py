import torch
from diffusers import StableDiffusionPipeline

# Định nghĩa tham số
rand_seed = torch.manual_seed(42)
NUM_INFERENCE_STEPS = 100
GUIDANCE_SCALE = 0.85
HEIGHT = 512
WIDTH = 512

# Danh sách model
model_list = ["nota-ai/bk-sdm-small",
              "CompVis/stable-diffusion-v1-4",
              "runwayml/stable-diffusion-v1-5",
              "prompthero/openjourney",
              "stabilityai/stable-diffusion-xl-base-1.0",
              "stabilityai/stable-diffusion-3.5-large",
              "shuttleai/shuttle-3-diffusion"
              ]


def create_pipeline(model_name = model_list[6]):
    # Nếu máy có GPU CUDA
    if torch.cuda.is_available():
        print("Using GPU")
        print(model_name)
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype = torch.float16,
            use_safetensors = True
        ).to("cuda")
    else:
        print("Using CPU")
        print(model_name)
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
    return pipeline

def text2img(prompt, pipeline):
    images = pipeline(
        prompt,
        guidance_scale = GUIDANCE_SCALE,
        num_inference_steps = NUM_INFERENCE_STEPS,
        generator = rand_seed,
        num_images_per_request = 1,
        height = HEIGHT,
        width = WIDTH
    ).images

    return images[0]
