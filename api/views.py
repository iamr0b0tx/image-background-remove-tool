import os, gdown

from PIL import Image
from io import BytesIO
from .model import U2NET
from skimage import transform

from rest_framework import status
from django.http import Http404, HttpResponse
from rest_framework.decorators import api_view

from django.core.files.uploadedfile import InMemoryUploadedFile

# u2net
model_name = "u2net"
model_dir = os.path.join("models", model_name)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "u2net.pth")
model_url = "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"

if not os.path.exists(model_path):
    gdown.download(model_url, model_path)

model = U2NET()

@api_view(('POST',))
def remove_bg(request):
    resolution = request.data.get("resolution", None)
    image_object = request.FILES.get("image", None)
    
    try:
        if resolution is None or image_object is None:
            raise AttributeError

        # load bytes
        image = image_object.read()

        # processs
        image = model.process_image(image, resolution=resolution)
        if image is None:
            raise AttributeError
        
        return HttpResponse(image, content_type="image/png")

    except (IOError, AttributeError) as e:
        print(e)
        raise Http404("Invalid Image!")

@api_view(('POST',))
def replace_bg(request):
    image = request.FILES.get("image", None)
    background = request.FILES.get("background", None)
    
    try:
        if image is None or background is None:
            raise AttributeError

        # load bytes
        image = Image.open(BytesIO(image.read()))
        background = Image.open(BytesIO(background.read()))

        print(image.size)
        print(background.size)
        image_width, image_height = image.size
        background_width, background_height = background.size

        scale = max(1, image_height/background_height, image_width/background_width)

        background_width = (background_width * scale)
        background_height = (background_height * scale)
        background = background.resize((background_width, background_height), Image.ANTIALIAS)

        print(background_width, background_height)

        left = .5 * (background_width - image_width)
        right = left + image_width
        top = .5 * (background_height - image_height)
        bottom = top + image_height

        print(left, top, right, bottom)
        background = background.crop((left, top, right, bottom))

        new_image = Image.new('RGBA', background.size, (0, 0, 0, 0))
        new_image.paste(background, (0, 0))
        new_image.paste(image, (0, 0), mask=image)

        new_image_io = BytesIO()
        new_image.save(new_image_io, format='PNG')
        new_image_bytes = InMemoryUploadedFile(new_image_io, None, 'result.png', 'image/png', new_image_io.tell, None)
        
        return HttpResponse(new_image_bytes, content_type="image/png")

    except (IOError, AttributeError) as e:
        print(e)
        raise Http404("Invalid Image!")
