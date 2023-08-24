var chartDom = document.getElementById('Chart');
var myChart = echarts.init(chartDom);


// var datesDataAttr = chartDom.getAttribute('data-dates');
// // 'data-dates' 속성 값이 있는 경우에만 JSON 파싱하여 사용
// var dates_data;
// var avg_deltaDataAttr = chartDom.getAttribute('avg-delta-data');
// // 'data-dates' 속성 값이 있는 경우에만 JSON 파싱하여 사용
// var avg_delta_data;
var datesDataAttr = chartDom.getAttribute('data-dates');
var avg_deltaDataAttr = chartDom.getAttribute('avg-delta-data');
var dates_data = datesDataAttr;  // JSON.parse를 사용하지 않음
var avg_delta_data = avg_deltaDataAttr;  // JSON.parse를 사용하지 않음
console.log("dates_data:", dates_data);
console.log("avg_delta_data:", avg_delta_data);


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
