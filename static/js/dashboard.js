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
// function DragDrop(){
//   const dropbox = document.querySelector('.file_box');
//   const imagebox = document.querySelector('.inner_box')
//   const input_filename = document.querySelector('.file_name');

//   //박스 안에 drag 하고 있을 때
//   dropbox.addEventListener('dragover', function (e) {
//     e.preventDefault();
//     this.style.backgroundColor = 'rgb(13 110 253 / 25%)';
//   });

//   //박스 밖으로 drag가 나갈 때
//   dropbox.addEventListener('dragleave', function (e) {
//     this.style.backgroundColor = 'white';
//   });

//   //박스 안에 drop 했을 때
//   dropbox.addEventListener('drop', function (e) {
//     e.preventDefault();
//     this.style.backgroundColor = 'white';

//     //파일 이름을 text로 표시
//     let filename = e.dataTransfer.files[0].name;
//     imagebox.style.display = "none";
//     input_filename.innerHTML = filename;
//   });
// };

// DragDrop();

$(document).ready(function(){

  initFileUploader("#zdrop");

  function initFileUploader(target) {
    var previewNode = document.querySelector("#zdrop-template");
    previewNode.id = "";
    var previewTemplate = previewNode.parentNode.innerHTML;
    previewNode.parentNode.removeChild(previewNode);


    var zdrop = new Dropzone(target, {
      // url: '/Home/UploadFile',
      maxFilesize:20,
      previewTemplate: previewTemplate,
      autoQueue: true,
      previewsContainer: "#previews",
      clickable: "#upload-label"
    });

    zdrop.on("addedfile", function(file) { 
      $('.preview-container').css('visibility', 'visible');
    });

    zdrop.on("totaluploadprogress", function (progress) {
      var progr = document.querySelector(".progress .determinate");
      if (progr === undefined || progr === null)
        return;

      progr.style.width = progress + "%";
    });

    zdrop.on('dragenter', function () {
      $('.fileuploader').addClass("active");
    });

    zdrop.on('dragleave', function () {
      $('.fileuploader').removeClass("active");			
    });

    zdrop.on('drop', function () {
      $('.fileuploader').removeClass("active");	
    });
    
    var toggle = true;
    /* Preview controller of hide / show */
    $('#controller').click(function() {
      if(toggle){
        $('#previews').css('visibility', 'hidden');
        $('#controller').html("keyboard_arrow_up");
        $('#previews').css('height', '0px');
        toggle = false;
      }else{
        $('#previews').css('visibility', 'visible');
        $('#controller').html("keyboard_arrow_down");
        $('#previews').css('height', 'initial');
        toggle = true;
      }
    });
  }

});

ShowChart();