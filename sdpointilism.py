#!/usr/bin/python3
from PIL import Image
from random import randint
from os.path import exists
from diffusers import StableDiffusionImg2ImgPipeline
import torch
import shutil

in_file="0.png"
#infile expected resolution 4096 × 2416
IMAGES_FOLDER="img"
OUTPUT_FOLDER="out"
TMP_FOLDER="tmp"
iters=100000
sd_width=512
sd_height=512
count=0
prompt="isometric city extremely detailed simcity, colorful, fantasy, particle lights, boldly outlined, high resolution, uhd"
steps=50
strength=0.3

def get_image_size():
    im = Image.open(IMAGES_FOLDER+'/'+in_file)
    return im.size

def dummy(images, **kwargs):
    return images, False

def gen_frames(count):
    while count <= iters:
        if not exists(OUTPUT_FOLDER+'/'+str(count+1)+'.png'):
            img=Image.open(OUTPUT_FOLDER+'/'+str(count)+'.png')
            img=img.convert('RGB')
            x=randint(1,im_width-sd_width)
            y=randint(1,im_height-sd_height)
            print('{} {} {} {}'.format(im_width,im_height,x,y))
            n_img=img.crop((x,y,x+sd_width,y+sd_height))
            n_img.save(TMP_FOLDER+'/'+str(count)+'.png')
            torch.cuda.empty_cache()
            outpainted_image = pipe(prompt=prompt, image=n_img, strength=strength,num_inference_steps=steps ).images[0]
            outpainted_image.save('tmp/'+str(count)+'n.png')
            img.paste(outpainted_image,(x,y))
            count=count+1
            img.save(OUTPUT_FOLDER+'/'+str(count)+'.png')
        else:
            print('File {} already exists. Skipping to next.'.format(OUTPUT_FOLDER+'/'+str(count+1)+'.png'))
            count=count+1

(im_width,im_height)=get_image_size()
#pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("XpucT/Deliberate", torch_dtype=torch.float16)

pipe.safety_checker = dummy
pipe = pipe.to("cuda")
shutil.copyfile(IMAGES_FOLDER+'/'+in_file, OUTPUT_FOLDER+'/'+in_file)
gen_frames(count)
