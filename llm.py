from base64 import b64encode
import openai
from openai import OpenAI
import os
import json

IMG_QUALITY = "high"
MAX_TOKENS = 16384

# ----------------- Image encoding -----------------
def encode_image(image_path) -> str:
	"""Encode image to base64."""
	with open(image_path, mode="rb") as img_file:
		return b64encode(img_file.read()).decode("utf-8")

def encode_few_shot_samples(few_shot_mappings):
    """Encode few-shot images to base64."""

    return {
		# possible formats:
		# label: list of paths
		# description: single path
        key: [encode_image(str(p)) for p in paths] if isinstance(paths, list) else [encode_image(str(paths))]
        for key, paths in few_shot_mappings.items()
    }
# -------------------------------------------------

# ------------------- LLM client ------------------
def initialize_client(model_name):
    """Initialize the appropriate client based on model name."""
    if "gpt" in model_name or "o4" in model_name:
        client = OpenAI()
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        # if lmstudio is used from windows and wsl - get url with "ip route/"
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lmstudio")
    
    return client
# -------------------------------------------------

# ---------------- Process messages ---------------
def few_shot(
	messages: dict,
	img_quality: str,
	few_shot_samples: dict,
	interleave: bool = True
):
	"""Generate few-shot examples messages."""
	if not interleave:
		for instruct, shot_images in few_shot_samples.items():
			image_content = [
				{
					"type": "image_url",
					"image_url": {
						"url": f"data:image/jpeg;base64,{image}",
						"detail": f"{img_quality}",
					}
				}
				for image in shot_images
			]
			examples = {
				"role": "user",
				"content": [{"type": "text", "text": instruct}, *image_content],
			}

			messages.append(examples)
	if interleave:
		# Organize samples by class
		organized_samples = {}
		for instruct, shot_images in few_shot_samples.items():
			organized_samples[instruct] = shot_images
		
		# Get all instruction types
		instruction_types = list(organized_samples.keys())
		
		# Find the maximum number of samples in any class
		max_samples = max(len(images) for images in organized_samples.values())
		
		# Interleave samples
		for i in range(max_samples):
			for instruct in instruction_types:
				# Skip if we've used all samples for this class
				if i >= len(organized_samples[instruct]):
					continue
				
				# Create a message with one instruction and one image
				image = organized_samples[instruct][i]
				example = {
					"role": "user",
					"content": [
						{"type": "text", "text": instruct},
						{
							"type": "image_url",
							"image_url": {
								"url": f"data:image/jpeg;base64,{image}",
								"detail": f"{img_quality}",
							}
						}
					]
				}
				messages.append(example)

	return messages

def _gen_system_message(system_prompt: str):
	"""Generate system message."""
	return {
		"role": "system",
		"content": [
			{
				"type": "text",
				"text": system_prompt,
			},
		],
	}

def _gen_user_message(user_query: str, image: str, img_quality: str):
	"""Generate user message."""
	return {
		"role": "user",
		"content": [
            {
                "type": "text",
                "text": user_query,
            },
            *[
                {
                    "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                        "detail": f"{img_quality}",
                    }
                }
            ],
        ],
    }

def process_messages(system_prompt, user_query, query_img, few_shot_samples=None):
	"""Create messages that will be sent to the model."""
	# System prompt
	messages = [_gen_system_message(system_prompt)]

	if few_shot_samples:

		# First part of user message: instructions and few-shot examples
		user_query_pre = user_query.split("-----------")[0].strip()
		messages.append(
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": user_query_pre,
					}
				],
			}
		)
	
		messages = few_shot(
			messages=messages,
			img_quality=IMG_QUALITY,
			few_shot_samples=few_shot_samples,
		)
		message_for_sample = messages.copy()
			
		# Second part of user message: instruction and image
		user_query_post = user_query.split("-----------")[1].strip()
		message_for_sample.append(_gen_user_message(user_query_post, query_img, IMG_QUALITY))
	else:
		# user message and image
		message_for_sample = messages.copy()
		message_for_sample.append(_gen_user_message(user_query, query_img, IMG_QUALITY))

	return message_for_sample
# -------------------------------------------------

# ------------------ Model prediction -------------
def get_model_prediction(client, model_name, message_for_sample):

	if "gpt" in model_name:
		character_schema = {"type": "json_object"}
	else:
		character_schema = {
			"type": "json_schema",
			"json_schema": {
				"name": "character",
				"schema": {
					"type": "object",
					"properties": {
						"thoughts": {"type": "string"},
					"answer": {"type": "string"},
					"score": {"type": "number"}
				},
					"required": ["thoughts", "answer", "score"]
				}
			}
		}

	"""Get prediction from the model."""
	
	response = client.chat.completions.create(
		model=model_name,
		messages=message_for_sample,
		#max_tokens=MAX_TOKENS,
		seed=42,
		temperature=1,
		response_format=character_schema
	)
	
	return json.loads(response.choices[0].message.content)
# -------------------------------------------------