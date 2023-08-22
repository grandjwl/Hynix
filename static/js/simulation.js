// 대시보드 showing 함수
function ShowChart() {
  const table = document.querySelector(".table-responsive"),
    btn1 = table.querySelector(".reupload"),
    btn2 = table.querySelector(".pred");
  var chartDom = document.getElementById('chart');
  btn2.onclick = () => {
    btn1.style.display = "none";
    btn2.style.display = "none";
    chartDom.style.display = "block";
  };

  
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

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// file upload form
$('#chooseFile').bind('change', function () {
  var filename = $("#chooseFile").val();
  if (/^\s*$/.test(filename)) {
    $(".file-upload").removeClass('active');
    $("#noFile").text("No file chosen..."); 
  }
  else {
    $(".file-upload").addClass('active');
    $("#noFile").text(filename.replace("C:\\fakepath\\", "")); 
  }
});

// drag&drop file upload
function DragDrop() {
  const dropArea1 = document.querySelector(".drop_box1"),
    form = dropArea1.querySelector("#uploadForm"),
    area = document.querySelector(".file_upload_container"),
    table = document.querySelector(".table-responsive");

  form.addEventListener("submit", function (e) {
    e.preventDefault();

    // 폼이 제출될 때의 작업을 수행합니다.
    table.style.display = "block";
  });
};
// showing table after upload
// function ShowTable() {
//   const table = document.querySelector(".table-responsive"),
//     area = document.querySelector(".file_upload_container"),
//     box = document.querySelector(".drop_box2"),
//     btn = box.querySelector(".upload_btn");

//     btn.onclick = () => {
//       area.style.display = "none";
//       table.style.display = "block";
//     };
// }

// showing file upload ui
function ShowFileUpload() {
  const table = document.querySelector(".table-responsive"),
    btn = table.querySelector(".reupload"),
    area = document.querySelector(".file_upload_container");

  btn.onclick = () => {
    table.style.display = "none";
    area.style.display = "block";
  };

}

// ShowTable();
ShowChart();
ShowFileUpload();
DragDrop();