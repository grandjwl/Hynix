// 대시보드 showing 함수
function ShowChart() {
  const table = document.querySelector(".table-responsive"),
    btn2 = table.querySelector(".pred");
  var chartDom = document.getElementById('chart');
  btn2.onclick = () => {
    btn2.style.display = "none";
    chartDom.style.display = "block";
  };

  var myChart = echarts.init(chartDom);
  var option;

  var pred_max = Array(minmax.max);
  var pred_min = Array(minmax.min);
  const diff = pred_max.map((x, y) => x - pred_min[y]);
  console.log(diff);

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
      data: ["x1"]
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

// file upload form
function UploadFile() {
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
};

// showing table
function FillTable() {
  // 전체 데이터의 첫 번째 id값 저장
  const dataFirstKey = Object.keys(data)[0];
  // 데이터의 컬럼 정보 추출
  const valKeys = Object.keys(data[dataFirstKey]);

  // 전체 데이터의 id값들 저장
  const dataKeys = Object.keys(data);

  const table = document.querySelector(".table");
  const theadr = document.querySelector(".tableHead");
  const tbody = document.querySelector(".tableBody");

  // lot id 컬럼명 추가
  theadr.innerHTML += `
      <th>ID</th>
    `
  // id 제외한 컬럼명들 추가
  for(var key in valKeys){
    theadr.innerHTML += `
      <th>${valKeys[key]}</th>`
  }

  // 데이터 추가
  for(var key in dataKeys){
    const val = data[dataKeys[Number(key)]];
    var content = `
    <tr class="vals">
    <td>${key}</td>
    `;
    for(var v in val){
      if (!val[v]){
        content += `
        <td></td>
        `
      }
      else{
        content += `
        <td>${val[v]}</td>
        `
      }
    };
    content += `</tr>`
    tbody.innerHTML += content;
  };
  const area = document.querySelector(".table-responsive");
  area.style.display = "block";
};

UploadFile();
FillTable();
ShowChart();
