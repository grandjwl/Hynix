from django.shortcuts import render
# Create your views here.

def main(request):
    return render(request, 'hynix/main.html',{"contents":"<h1>main page</h1>"})

def simulation(request):
    return render(request, "hynix/simulation.html", {"contents":"<h1>simulator page</h1>"})

def lifecycle(request):
    return render(request, "hynix/model_lifecycle.html", {"contents":"<h1>mlc page</h1>"})