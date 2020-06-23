import os, gdown

from PIL import Image
from io import BytesIO
from .model import U2NET

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
    image_object = request.FILES.get("image", None)
    
    try:
        # load bytes
        image = image_object.read()
        # print(type(image))

        # processs
        image = model.process_image(image)
        
        return HttpResponse(image, content_type="image/png")

    except (IOError, AttributeError) as e:
        print(e)
        raise Http404("Invalid Image!")

@api_view(('POST',))
def replace_bg(request):
    image_object = request.FILES.get("image", None)
    
    try:
        # load bytes
        image = Image.open(BytesIO(image_object.read()))

        filename1 = 'bg.jpg'
        bg = Image.open(filename1, 'r')

        new_image = Image.new('RGBA', bg.size, (0, 0, 0, 0))
        new_image.paste(bg, (0, 0))
        new_image.paste(image, (0, 0), mask=image)

        new_image_io = BytesIO()
        new_image.save(new_image_io, format='PNG')
        new_image_bytes = InMemoryUploadedFile(new_image_io, None, 'result.png', 'image/png', new_image_io.tell, None)
        
        return HttpResponse(new_image_bytes, content_type="image/png")

    except (IOError, AttributeError) as e:
        print(e)
        raise Http404("Invalid Image!")
