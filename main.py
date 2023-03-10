from flask import Flask, jsonify, make_response, render_template, request, send_file
from itsdangerous import base64_encode
from jinja2 import Environment
import io
import base64
from diffusers import StableDiffusionImg2ImgPipeline
import torch
import requests
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import glob 
from typing import Optional, List, Union
#from flask_ngrok import run_with_ngrok
import redis
from redis import Redis
from rq import Queue
from rq.job import Job

# Connect to Redis instance
r = redis.Redis()
q = Queue(connection = r)
## Run redis-server in terminal
## Run rq worker in terminal

# Load Function
def multiple_rounds_img2img(
  init_image: Image,
  prompt: str,  
  negative_prompt: str,
  strength_array: List[float],
  guidance_array: Union[List[float], List[int]],
  final_images_to_return: Optional[int] = 5,
  num_rounds: Optional[int] = 4,
  seed: Optional[int] = 123) -> List:

  # Parameter checking
  ## init_image
  assert isinstance(init_image, Image.Image), "init_image must be an Image"

  ## prompt & negative_prompt
  assert isinstance(prompt, str) and len(prompt) > 0, "Prompt provided must be a comma separated string and cannot be an empty string" 
  assert isinstance(negative_prompt, str), "Negative Prompt provided must be a comma separated string"

  ## num rounds
  assert num_rounds > 0, "num_rounds must be greater than 0"

  ## strength_array & guidance array
  assert len(strength_array) == num_rounds, 'strength_array length must be identical to num_rounds'
  assert len(guidance_array) == num_rounds, 'guidance_array length must be identical to num_rounds'

  ## final_images_to_return
  assert final_images_to_return > 0, "final_images_to_return must be greater than 0"

  ## seed
  assert isinstance(seed, int), "seed must be an integer"
  
  # Main Body
  torch.manual_seed(seed)
  output_image_array = [init_image]

  for idx in list(range(0, num_rounds - 1)):
    
    img2imgpipeline = img2imgpipe(prompt = prompt,
                          image=output_image_array[idx],
                          strength=strength_array[idx],
                          guidance_scale=guidance_array[idx],
                          num_inference_steps=400,
                          num_images_per_prompt = 1,
                          negative_prompt = negative_prompt)

    output_image_array.append( img2imgpipeline.images[0] )

    # For final round of inference
    torch.manual_seed(seed)
    img2imgpipeline_final = img2imgpipe(prompt = prompt,
                            image=output_image_array[-1],
                            strength=strength_array[-1],
                            guidance_scale=guidance_array[-1],
                            num_inference_steps=400,
                            num_images_per_prompt = final_images_to_return,
                            negative_prompt = negative_prompt)

    return img2imgpipeline_final.images


#??Load model
img2imgpipe = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/mo-di-diffusion", torch_dtype=torch.float16)
#img2imgpipe.to("cuda")

def queue_image_generator(initial_image_str: str, prompt: str, negative_prompt: str, final_images_to_return: int = 1, seed: int = 123):
    
    # Load the image data into a PIL Image object
    img = Image.open(io.BytesIO(initial_image_str))
    
    returned_imgs = multiple_rounds_img2img(
    init_image = img,
    prompt = prompt,
    negative_prompt = negative_prompt,
    strength_array = [0.7],
    guidance_array = [20.0],
    final_images_to_return = final_images_to_return,
    num_rounds = 1,
    seed = seed)

    #??Convert generated image to bytestring
    buffered = io.BytesIO()
    returned_imgs[0].save(buffered, format = "JPEG")
    returned_imgs_data = buffered.getvalue()
    returned_imgs_b64 = base64.b64encode(returned_imgs_data).decode('utf-8')

    response_data = {"generated_image": returned_imgs_b64}

    json_response = jsonify(response_data)

    return json_response


# Set flask app and set to ngrok
app = Flask(__name__)
#run_with_ngrok(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def generate_image():
    
    # Get Prompt
    #prompt = request.form['prompt-input']
    
    # get uploaded file
    uploaded_file = request.files['file']

    # read image and rotate
    img = Image.open(uploaded_file)
    
    returned_imgs = multiple_rounds_img2img(
    init_image = img,
    prompt = "a stuffed brown meerkat dressed in a zebra suit, cartoon, Pixar, Disney character, 3D render, modern disney style",
    negative_prompt = "disfigured, misaligned, ugly, blurry, grumpy, grey, dark, big eyes, person, human, fuzzy, furry",
    strength_array = [0.7, 0.6, 0.5, 0.4],
    guidance_array = [20.0, 18.0, 16.0, 14.0],
    final_images_to_return = 5,
    num_rounds = 4,
    seed = 123)

    # Returned Images as bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    original_bytes = base64.b64encode(img_bytes.getvalue())
    
    img_bytes = io.BytesIO()
    returned_imgs[0].save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    img_0_bytes = base64.b64encode(img_bytes.getvalue())

    img_bytes = io.BytesIO()
    returned_imgs[1].save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    img_1_bytes = base64.b64encode(img_bytes.getvalue())

    img_bytes = io.BytesIO()
    returned_imgs[2].save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    img_2_bytes = base64.b64encode(img_bytes.getvalue())

    img_bytes = io.BytesIO()
    returned_imgs[3].save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    img_3_bytes = base64.b64encode(img_bytes.getvalue())
    
    img_bytes = io.BytesIO()
    returned_imgs[4].save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    img_4_bytes = base64.b64encode(img_bytes.getvalue())


    return render_template('result.html', 
                           original=original_bytes.decode('utf-8'), 
                           img_0=img_0_bytes.decode('utf-8'), 
                           img_1=img_1_bytes.decode('utf-8'), 
                           img_2=img_2_bytes.decode('utf-8'),
                           img_3=img_3_bytes.decode('utf-8'),
                           img_4=img_4_bytes.decode('utf-8'),
                           )






@app.route('/generate_image_rest', methods=['POST'])
def generate_image_rest():

    if request.is_json:
        
        # Get the JSON data from the POST request
        data = request.json

        # Get the base64-encoded image data and decode it
        image_b64 = data.get('initial_image', '')
        image_data = base64.b64decode(image_b64)

        #??Get parameters
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative', '')
        final_images_to_return = int(data.get('final_images_to_return', ''))
        seed = int(data.get('seed', ''))
        
        task = q.enqueue(queue_image_generator, image_data, prompt, negative_prompt, final_images_to_return, seed)
        # return the task ID to the client
        return jsonify({'task_id': task.id})
    
    else: 
        
        return make_response(jsonify({"message": "No JSON received"}), 400)



@app.route('/get_result/<task_id>')
def get_result(task_id):
    task = q.fetch_job(task_id)
    if task.is_finished:
        return jsonify(task.result)
    else:
        return jsonify({'status': 'in progress'})


if __name__ == '__main__':
    app.run(debug=True)
