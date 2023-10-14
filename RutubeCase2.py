!pip install transformers
from transformers import pipeline, AutoTokenizer, EncoderDecoderModel, BlipProcessor, BlipForConditionalGeneration

import torch

!pip install sentencepiece
import sentencepiece

!pip install --upgrade diffusers[torch]
from diffusers import DiffusionPipeline, StableDiffusionPipeline

import pandas as pd

from os import listdir
from os.path import isfile, join

from PIL import Image
import os
import cv2



#summarizing model
tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/rubert_telegram_headlines", do_lower_case=False, do_basic_tokenize=False, strip_accents=False)
model = EncoderDecoderModel.from_pretrained("IlyaGusev/rubert_telegram_headlines", device = torch.device('cuda'))

#translating model
tokenizer_tr = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
tr_pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en", tokenizer = tokenizer_tr, device = torch.device('cuda'))

#create descriptions for images
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model_desc_for_img = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

#summary for descroptions for images
pipe = pipeline("summarization", model="facebook/bart-large-cnn")

#generating model ANIME STYLE
img_gen_3 = DiffusionPipeline.from_pretrained("animelover/novelai-diffusion", custom_pipeline="lpw_stable_diffusion", torch_dtype=torch.float16, use_auth_token="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")
img_gen_3.safety_checker = None # we don't need safety checker. you can add not safe words to negative prompt instead.
img_gen_3 = img_gen_3.to("cuda")


def generate_video_preview(video, author_comments = None, tags = None): # -> List[BytesIO])  # Список сгенерированных превью-картинок

  ##create a folder to store extracted images

  folder = 'images_from_video'  
  os.mkdir(folder)

  vidcap = cv2.VideoCapture(video)
  count = 0
  while True:
      success,image = vidcap.read()
      if not success:
          break
      cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), image)     # save frame as JPEG file
      count += 1
  
  # create descriptions for images
  file_names = [f for f in listdir(f'{folder}') if isfile(join(f'{folder}', f))]
  file_names.sort()
  indx = []
  for i in range(0, len(file_names), 25):
    indx.append(i)
  
  img_description = []
  for file in indx:
    img = Image.open(f'{folder}/frame{file}.jpg').convert('RGB')

    inputs = processor(img, return_tensors="pt").to("cuda")

    out = model_desc_for_img.generate(**inputs)
    img_description.append(processor.decode(out[0], skip_special_tokens=True))

  #concat descriptions
  text = ' '.join(img_description)
  
  #summary
  video_summ = pipe(text[0:1000], max_length=130, min_length=30, do_sample=False)



  #summary author_comments
  if author_comments is not None:

    if len(author_comments)>35:
      article_text = author_comments

      input_ids = tokenizer(
      [article_text],
      add_special_tokens=True,
      max_length=256,
      padding="max_length",
      truncation=True,
      return_tensors="pt",
      )["input_ids"]

      output_ids = model.generate(
      input_ids=input_ids,
      max_length=100,
      no_repeat_ngram_size=3,
      num_beams=10,
      top_p=0.95
      )[0]

      author = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True, device = torch.device('cuda'))
      #translate rus-eng
      tr_author = tr_pipe(author)

    elif len(author_comments)<=35:
      tr_author = tr_pipe(author_comments)
  elif author_comments is None:
    tr_author = tr_pipe('картинка')

  if tags is not None:
    tags = tr_pipe(tags)
  elif tags is None:
    tags = tr_pipe('обложка')

  #make a prompt from summaries and tags
  for x in video_summ[0].values():
    prompt1 = str(x)
  for x in tr_author[0].values():
    prompt2 = str(x)
  for x in tags[0].values():
    prompt3 = str(x)
  prompt = " ".join([prompt1, prompt2, prompt3])

  #generate image ANIME STYLE
  neg_prompt = "lowres, bad anatomy, error body, error hair, error arm, error hands, bad hands, error fingers, bad  fingers, missing fingers, error legs, bad legs, multiple legs, missing legs, error lighting, error shadow, error reflection, text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
  # we don't need autocast here, because autocast will make speed slow down.
  image = img_gen_3.text2img(prompt,negative_prompt=neg_prompt, width=512,height=768,max_embeddings_multiples=5,guidance_scale=12).images[0]
  image.save("video.png")

  #convert from *.png to BytesIO:
  with open('video.png', 'rb') as f:
    video_byte = f.read()


  return video_byte



def generate_avatar_photo(avatar, avatar_description = None): # -> List[BytesIO]:  # Список сгенерированных аватарок
  
	#foto with style

	#summary avatar_description
  if avatar_description is not None:

    if len(avatar_description)>35:
      article_text = avatar_description

      input_ids = tokenizer(
      [article_text],
      add_special_tokens=True,
      max_length=256,
      padding="max_length",
      truncation=True,
      return_tensors="pt",
      )["input_ids"]

      output_ids = model.generate(
      input_ids=input_ids,
      max_length=100,
      no_repeat_ngram_size=3,
      num_beams=10,
      top_p=0.95
      )[0]

      avatar = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True, device = torch.device('cuda'))
      #translate rus-eng
      tr_avatar = tr_pipe(avatar)

    elif len(avatar_description)<=35:
      tr_avatar = tr_pipe(avatar_description)
  elif avatar_description is None:
    tr_avatar = tr_pipe('картинка')


  prompt = [str(x) for x in tr_avatar[0].values()]


  #generate image ANIME STYLE
  neg_prompt = "lowres, bad anatomy, error body, error hair, error arm, error hands, bad hands, error fingers, bad  fingers, missing fingers, error legs, bad legs, multiple legs, missing legs, error lighting, error shadow, error reflection, text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
  # we don't need autocast here, because autocast will make speed slow down.
  image = img_gen_3.text2img(prompt,negative_prompt=neg_prompt, width=512,height=768,max_embeddings_multiples=5,guidance_scale=12).images[0]
  image.save("avatar.png")


  #convert from *.png to BytesIO:
  with open('avatar.png', 'rb') as f:
    avatar_byte = f.read()


  return avatar_byte





def generate_channel_background_image(channel_background_image_description = None): # -> List[BytesIO]:  # Список сгенерированных задних фонов для канала
   #summary generate_channel_background_image
  if channel_background_image_description is not None:

    if len(channel_background_image_description)>35:
      article_text = channel_background_image_description

      input_ids = tokenizer(
      [article_text],
      add_special_tokens=True,
      max_length=256,
      padding="max_length",
      truncation=True,
      return_tensors="pt",
      )["input_ids"]

      output_ids = model.generate(
      input_ids=input_ids,
      max_length=100,
      no_repeat_ngram_size=3,
      num_beams=10,
      top_p=0.95
      )[0]

      background = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True, device = torch.device('cuda'))
      #translate rus-eng
      tr_background = tr_pipe(background)

    elif len(channel_background_image_description)<=35:
      tr_background = tr_pipe(channel_background_image_description)
  elif channel_background_image_description is None:
    tr_background = tr_pipe('фон')


  prompt = [str(x) for x in tr_background[0].values()]

   #generate 3 image ANIME STYLE
  neg_prompt = "lowres, bad anatomy, error body, error hair, error arm, error hands, bad hands, error fingers, bad  fingers, missing fingers, error legs, bad legs, multiple legs, missing legs, error lighting, error shadow, error reflection, text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
  # we don't need autocast here, because autocast will make speed slow down.
  image = img_gen_3.text2img(prompt,negative_prompt=neg_prompt, width=512,height=768,max_embeddings_multiples=5,guidance_scale=12).images[0]
  image.save("background.png")

  #convert from *.png to BytesIO:
  with open('background.png', 'rb') as f:
    background_byte = f.read()

  return background_byte



def choose_cover_from_video (video):
  folder = 'images_from_video'  
  os.mkdir(folder)

  vidcap = cv2.VideoCapture(video)
  count = 0
  while True:
      success,image = vidcap.read()
      if not success:
          break
      cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), image)     # save frame as JPEG file
      count += 1
  
  # create descriptions for images
  file_names = [f for f in listdir(f'{folder}') if isfile(join(f'{folder}', f))]
  file_names.sort()
  indx = []
  for i in range(0, len(file_names), 25):
    indx.append(i)
  
  img_description = []
  for file in indx:
    img = Image.open(f'{folder}/frame{file}.jpg').convert('RGB')

    inputs = processor(img, return_tensors="pt").to("cuda")

    out = model_desc_for_img.generate(**inputs)
    img_description.append(processor.decode(out[0], skip_special_tokens=True))

  #concat descriptions
  text = ' '.join(img_description)
  
  #summary
  video_summ = pipe(text[0:1000], max_length=130, min_length=30, do_sample=False)
  for x in video_summ[0].values():
    prompt = str(x)

  dictance_with_prompt = []
  for sentence in img_description:
    dictance_with_prompt.append(distance(prompt.split(), sentence.split()))

  d = {'indx_video': indx, 'img_description': img_description, 'distance' : dictance_with_prompt}
  df = pd.DataFrame(data=d)
  df['indx_video'][df['distance'] == min(dictance_with_prompt)]

  for i in df['indx_video'][df['distance'] == min(dictance_with_prompt)]:
    with open(f'{folder}/frame{i}.jpg', 'rb') as f:
      cover = f.read()
  
  return cover
