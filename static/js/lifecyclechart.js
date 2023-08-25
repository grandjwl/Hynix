var chartDom = document.getElementById('Chart');
var myChart = echarts.init(chartDom);
console.log(chartDom)


// 속성 값 가져오기
var avgDeltaData = chartDom.getAttribute('avg_delta');
var datesData = chartDom.getAttribute('Date');
var dates_data = avgDeltaData.split(','); // 쉼표로 구분된 값들을 리스트로 변환
var avg_delta_data = datesData.split(','); // 쉼표로 구분된 값들을 리스트로 변환
console.log(dates_data);
console.log(avg_delta_data);
// var dates_data = ['2023-08-01', '2023-08-02', '2023-08-03', '2023-08-04', '2023-08-05'];
// var avg_delta_data = [10, 15, 8, 12, 18];



var option;
var option = {
  xAxis: {
    type: 'category',
    data: dates_data,  // 변경된 부분
    name: 'Date',
    nameTextStyle: {
      color: 'black',
      fontSize: 20,
      fontWeight: 'bold'
    },
    axisLabel: {
      formatter: '{value}',
      rotate: 45,
      textStyle: {
        color: 'black',
        fontSize: 15,
        fontWeight: 'bold',
      }
    }
  },



  yAxis: {
    type: 'value',
    name: 'Delta',
    nameTextStyle: {
      color: 'black',
      fontSize: 20,
      fontWeight: 'bold'},

    axisLabel: {
      show: true,
      textStyle: {
        fontWeight: 'bold',
        color: 'black',
        fontSize: 15,
      },
    },
  },



  series: [
    {
      data: avg_delta_data, 
      type: 'line',
      lineStyle: {
        color: 'blue',
      },
      markLine: {
        data: [
          {
            yAxis: 5,
            label: {
              show: true,
              fontWeight: 'bold',
              color: 'red',
              formatter: 'Threshold'
            }
          }
        ],
        lineStyle: {
          color: 'red'
        }
      }
    }
  ],
};


myChart.setOption(option);
