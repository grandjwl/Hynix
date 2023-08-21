from django.shortcuts import render
# Create your views here.

def main(request):
    return render(request, 'hynix/main.html',{"contents":"<h1>main page</h1>"})

def simulation(request):
    return render(request, "hynix/simulation.html", {"contents":"<h1>simulator page</h1>"})



# def lifecycle(request):
#     return render(request, "hynix/model_lifecycle.html")

def lifecycle(request):
    data = []
    for i in range(1, 101):
        datetime_value = f"2023-08-{i:02d} {i:02d}:00:00"
        lot_id = 1000 + i
        pred = 70 + i
        real = None  # 빈 값으로 설정
        data.append({"datetime": datetime_value, "lot_id": lot_id, "pred": pred, "real": real})
    
    return render(request, "hynix/lifecycle.html", {"data": data})








