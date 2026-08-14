from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse, HttpResponse

from .forms import ImageUploadForm
from .ml.predict import predict_image
from .models import Prediction
from .pdf_report import create_pdf


def home(request):

    prediction = None
    confidence = None
    probabilities = None
    image_url = None

    if request.method == "POST":

        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():

            image = form.cleaned_data["image"]

            fs = FileSystemStorage()

            filename = fs.save(image.name, image)

            image_path = fs.path(filename)

            image_url = fs.url(filename)

            prediction, confidence, probabilities = predict_image(image_path)

            Prediction.objects.create(
                image=filename,
                prediction=prediction,
                confidence=confidence
            )

            print("Prediction:", prediction)
            print("Confidence:", confidence)
            print("Saved to Database Successfully!")

    else:

        form = ImageUploadForm()

    history = Prediction.objects.order_by("-created_at")

    return render(
        request,
        "index.html",
        {
            "form": form,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "image_url": image_url,
            "history": history,
        },
    )
    def history(request):
        predictions = Prediction.objects.all().order_by('-created_at')

    return render(
        request,
        "history.html",
        {
            "predictions": predictions
        }
    )


def download_report(request):

    latest = Prediction.objects.last()

    if latest is None:
        return HttpResponse("No prediction found.")

    pdf_path = "Brain_Tumor_Report.pdf"

    create_pdf(
        pdf_path,
        latest.prediction,
        latest.confidence
    )

    return FileResponse(
        open(pdf_path, "rb"),
        as_attachment=True,
        filename="Brain_Tumor_Report.pdf"
    )

from .models import Prediction

def history(request):
    
    search = request.GET.get("search", "")

    if search:
        predictions = Prediction.objects.filter(
            prediction__icontains=search
        ).order_by("-created_at")
    else:
        predictions = Prediction.objects.all().order_by("-created_at")

    return render(
        request,
        "history.html",
        {
            "predictions": predictions,
            "search": search,
        },
    )