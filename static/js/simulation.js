// 대시보드 showing 함수
function ShowChart() {

  const table = document.querySelector(".table-responsive"),
    btn2 = table.querySelector(".pred");
  var chartDom = document.getElementById('myChart');
  btn2.onclick = () => {
    btn2.style.display = "none";
    chartDom.style.display = "block";
  };

  // var process = new Array();
  // var pred_max = new Array();
  // var pred_min = new Array();
  // var pred_avg = new Array();

  // var maxkey = Object.keys(minmax.max);
  // for (var i=1; i<maxkey.length; i+= 2) {
  //   pred_max.push(minmax.max[i]);
  //   pred_min.push(minmax.min[i]);
  //   pred_avg.push(minmax.avg[i]);
  //   process.push(minmax.process[i]);
  // }

  var process = ["x1","x2","x3","x4","x5"];
  var pred_max = [100,97,95,93,90];
  var pred_min = [65,68,70,75,80];
  var pred_avg = [70,75,79,82,85];

  console.log(pred_max);
  console.log(pred_min);
  console.log(pred_avg);
  console.log(process);

  var ctx = document.getElementById('myChart').getContext('2d');
  var chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: process,
      datasets: [
        {
          label: 'Min Values',
          data: pred_min,
          borderColor: 'red',
          fill: false
        },
        {
          label: 'Max Values',
          data: pred_max,
          borderColor: 'blue',
          fill: false
        },
        {
          label: 'Avg Values',
          data: pred_avg,
          backgroundColor: 'green', // 점의 배경색
          borderColor: 'green', // 점의 테두리 색
          pointRadius: 2, // 점의 반지름
          pointHoverRadius: 3, // 호버 시의 점의 반지름
          pointStyle: 'circle', // 점의 모양
          showLine: false, // 선 그래프 표시하지 않음
          fill: false
        }
      ]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true
        }
      }
    }
  });
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
