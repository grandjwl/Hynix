// 대시보드 showing 함수
function ShowChart() {
  const btn2 = document.querySelector(".pred");
  var chartDom = document.getElementById('myChart');
  var line = document.querySelector(".line");
  btn2.addEventListener('click',function() {
    line.style.display = "block";
    chartDom.style.display = "block";
  });

  var process = new Array();
  var pred_max = new Array();
  var pred_min = new Array();
  var pred_avg = new Array();

  var maxkey = Object.keys(minmax.max);
  for (var i=0; i<maxkey.length; i++) {
    pred_max.push(minmax.max[i]);
    pred_min.push(minmax.min[i]);
    pred_avg.push(minmax.avg[i]);
    process.push(minmax.process[i]);
  }
  var ymax = Math.max(pred_max);
  var ymin = Math.min(pred_min);

  var myChart = echarts.init(chartDom);
  var option;

  option = {
    title: {
      text: '반도체 수율 시뮬레이션 결과',
      left: 'center', 
    },
    xAxis: {
      type: 'category',
      name: "공정 진행 단계",
      data: process,
    },
    yAxis: {
      scale: true,
      name: "수율(%)",
    },
    series: [
      {
        data: pred_max,
        type: 'line',
        smooth: true
      },
      {
        data: pred_avg,
        type: 'scatter',
      },
      {
        data: pred_min,
        type: 'line',
        smooth: true
      }
    ],
    dataZoom: [
      {
        type: 'inside',
        xAxisIndex: [0],
        startValue: 0, // 배경 색 시작 지점
        endValue: process.length - 1, // 배경 색 종료 지점
        orient: 'horizontal',
        zoomLock: true, // 배경 색 영역을 잠금
        backgroundColor: 'rgba(1, 1, 1, 0)', // 배경 색
      }
    ]
  };
  option && myChart.setOption(option);
}

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


  const table = document.querySelector(".table-responsive");
  table.style.marginTop = "70px";
  const theadr = document.querySelector(".tableHead");
  const tbody = document.querySelector(".tableBody");

  // id 제외한 컬럼명들 추가
  for(var key in valKeys){
    theadr.innerHTML += `
      <th>${valKeys[key]}</th>`
  }

  // 데이터 추가
  for(var key in dataKeys){
    const val = data[dataKeys[Number(key)]];
    console.log(val);
    var content = `
    <tr class="vals">
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

function Toggle() {
  const toggleSwitch = document.getElementById("toggle-switch");
  toggleSwitch.addEventListener("change", function () {
      if (toggleSwitch.checked) {
          // ON일 때, value 값을 1로 설정
          toggleSwitch.value = "1";
          // statusElement.textContent = "YES";
      } else {
          // OFF일 때, value 값을 0으로 설정
          toggleSwitch.value = "0";
          // statusElement.textContent = "NO";
      }
  });
}

UploadFile();
Toggle();
ShowChart();
FillTable();


