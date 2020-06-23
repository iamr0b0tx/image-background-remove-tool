from django.shortcuts import render

# Create your views here.
def remove_bg(request):
    image = request.data

    print(image)

    return Response()
