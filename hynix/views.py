from django.shortcuts import render
# Create your views here.

def main(request):
    return render(request, 'hynix/main.html',{"contents":"<h1>main page</h1>"})

def simulation(request):
    return render(request, "hynix/simulation.html", {"contents":"<h1>simulator page</h1>"})



# def lifecycle(request):
#     return render(request, "hynix/model_lifecycle.html", {"contents":"mlc page"})

from datetime import date
from .forms import LifecycleForm

def lifecycle(request):
    if request.method == 'POST':
        form = LifecycleForm(request.POST)
        if form.is_valid():
            selected_date = form.cleaned_data['selected_date']
            simulation_value = form.cleaned_data['simulation_value']
            actual_value = form.cleaned_data['actual_value']
            # 추가적인 처리 로직을 여기에 작성하세요.
    else:
        form = LifecycleForm()

    context = {
        "contents": "mlc page",
        "form": form
    }
    return render(request, "hynix/model_lifecycle.html", context)



