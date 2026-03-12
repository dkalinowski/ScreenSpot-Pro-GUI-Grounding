import json
import os
import re

import base64
from io import BytesIO
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

import openai
from openai import BadRequestError

MODEL_NAME = os.environ.get("MODEL_NAME", "vlm")
BASE_URL = os.environ.get("OPENAI_API_BASE", "http://ov-ptl-13.sclab.intel.com:8181/v3")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "abc")
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", "99999999"))
MAX_TOKENS = int(os.environ.get("MAX_PIXELS", "10000"))


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


class Qwen3VLOVMSModel:
    def __init__(self, model_name=MODEL_NAME):
        if BASE_URL is not None:
            self.client = openai.OpenAI(
                api_key=OPENAI_KEY,
                base_url=BASE_URL,
            )
        else:
            self.client = openai.OpenAI(
                api_key=OPENAI_KEY,
            )
        self.model_name = model_name
        self.override_generation_config = {"temperature": 0.0}

    def load_model(self):
        print("Loading benchmark via OVMS!")

    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert("RGB")
        assert isinstance(image, Image.Image), "Invalid input image."

        # Calculate the real image size sent into the model
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=32,
            min_pixels=32 * 32,
            max_pixels=MAX_PIXELS,
        )
        print("Resized image size: {}x{}".format(resized_width, resized_height))
        resized_image = image.resize((resized_width, resized_height))

        base64_image = convert_pil_image_to_base64(resized_image)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant. The user will give you an instruction, and you MUST left click on the corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                },
                            },
                            {
                                "type": "text",
                                "text": instruction,
                            },
                        ],
                    },
                ],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "computer_use",
                        "description": f"Use a mouse to interact with a computer.\n* The screen's resolution is {resized_width}x{resized_height}.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. \n* You can only use the left_click action to interact with the computer.",
                        "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `left_click`: Click the left mouse button with coordinate (x, y).", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.", "type": "array"}, "required": ["action"], "type": "object"}},
                    },
                }],
                temperature=0.0,
                max_tokens=MAX_TOKENS,
            )
            message = response.choices[0].message
            response_text = message.content or ""
        except BadRequestError as e:
            print("OpenAI BadRequestError:", e)
            return None

        # Extract from OpenAI function call format (tool_calls)
        click_point = None
        bbox = None

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "computer_use":
                    try:
                        args = json.loads(tool_call.function.arguments)
                        coord = args.get("coordinate")
                        if coord and len(coord) == 2:
                            click_point = [float(coord[0]) / resized_width, float(coord[1]) / resized_height]
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        print("Failed to parse tool call arguments:", e)

        # Fallback: try extracting from text content
        if not click_point:
            bbox = extract_first_bounding_box(response_text)
            point = extract_first_point(response_text)
            if point:
                click_point = [point[0] / resized_width, point[1] / resized_height]

        print("------")
        print("Response text:", response_text)
        print("Tool calls:", message.tool_calls)
        print("Extracted point:", click_point)

        if not click_point and bbox:
            click_point = [((bbox[0] + bbox[2]) / 2) / resized_width, ((bbox[1] + bbox[3]) / 2) / resized_height]

        result_dict = {"result": "positive", "bbox": bbox, "point": click_point, "raw_response": response_text or str(message.tool_calls)}

        return result_dict

    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError()


def extract_first_bounding_box(text):
    # Regular expression pattern to match the first bounding box in the format [[x0,y0,x1,y1]]
    # This captures the entire float value using \d for digits and optional decimal points
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"

    # Search for the first match in the text
    match = re.search(pattern, text, re.DOTALL)

    if match:
        # Capture the bounding box coordinates as floats
        bbox = [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]
        return bbox
    return None


def extract_first_point(text):
    # Regular expression pattern to match the first point in the format [[x0,y0]]
    # This captures the entire float value using \d for digits and optional decimal points
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"

    # Search for the first match in the text
    match = re.search(pattern, text, re.DOTALL)

    if match:
        point = [float(match.group(1)), float(match.group(2))]
        return point

    return None
