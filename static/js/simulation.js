// 대시보드 showing 함수
function ShowChart() {
  var chartDom = document.getElementById('chart');
  var myChart = echarts.init(chartDom);
  var option;

  var pred_max = [99, 94, 90, 87, 85];
  var pred_min = [50, 60, 70, 80, 84];
  const diff = pred_max.map((x, y) => x - pred_min[y]);

  option = {
    title: {
      text: 'Accumulated Waterfall Chart'
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      },
      formatter: function (params) {
        let tar;
        if (params[1] && params[1].value !== '-') {
          tar = params[1];
        } else {
          tar = params[2];
        }
        return tar && tar.name + '<br/>' + tar.seriesName + ' : ' + tar.value;
      }
    },
    legend: {
      data: ['Max-Min']
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: ["x1","x2","x3","x4","x5"]
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 100
    },
    series: [
      {
        name: 'Min',
        type: 'bar',
        stack: 'Total',
        label: {
          show: true,
          position: 'insideTop'
        },
        silent: true,
        itemStyle: {
          borderColor: 'transparent',
          color: 'transparent'
        },
        emphasis: {
          itemStyle: {
            borderColor: 'transparent',
            color: 'transparent'
          }
        },
        data: pred_min
      },
      
      {
        name: 'Max-Min',
        type: 'bar',
        stack: 'Total',
        label: {
          show: true,
          position: 'inside'
        },
        data: diff
      },
      {
        name: 'Max',
        type: 'bar',
        stack: 'Total',
        label: {
          show: true,
          position: 'insideBottom',
        },
        silent: true,
        itemStyle: {
          borderColor: 'transparent',
          color: 'transparent'
        },
        emphasis: {
          itemStyle: {
            borderColor: 'transparent',
            color: 'transparent'
          }
        },
        data: pred_max
      }
    ]
  };

  option && myChart.setOption(option);
};

// drag&drop file upload
function DragDrop() {
  const dropArea1 = document.querySelector(".drop_box1"),
    dropArea2 = document.querySelector(".drop_box2"),
    button = dropArea1.querySelector(".choose_btn"),
    input = dropArea1.querySelector("input"),
    text = dropArea2.querySelector("h5");

  button.onclick = () => {
    input.click();
  };

  input.addEventListener("change", function (e) {
    dropArea1.style.display = "none";
    dropArea2.style.display = "block";

    text.style.display="none";
    var fileName = e.target.files[0].name;
    let filedata = `<h5><b>${fileName}</b></h5>`;
    document.getElementById("filename").innerHTML+=filedata;
  });
}

// showing table after upload
function ShowTable() {
  const table = document.querySelector(".table-responsive"),
    area = document.querySelector(".file_upload_container"),
    box = document.querySelector(".drop_box2"),
    btn = box.querySelector(".upload_btn");

    btn.onclick = () => {
      area.style.display = "none";
      table.style.display = "block";
    };
}
// side bar
// (() => {
//   'use strict'
//   const tooltipTriggerList = Array.from(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
//   tooltipTriggerList.forEach(tooltipTriggerEl => {
//     new bootstrap.Tooltip(tooltipTriggerEl)
//   })
// })()

ShowTable();
DragDrop();
ShowChart();