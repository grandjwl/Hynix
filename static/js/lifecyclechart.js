var chartDom = document.getElementById('Chart');
var myChart = echarts.init(chartDom);
console.log(chartDom)


var avg_delta_data = chartDom.getAttribute('avg-delta-data');
var dates_data = chartDom.getAttribute('data-dates');
// var avg_delta_data = JSON.parse(chartDom.getAttribute('avg-delta-data'));
// var dates_data = JSON.parse(chartDom.getAttribute('data-dates'));
console.log("avg_delta_data:", avg_delta_data);
console.log("dates_data:", dates_data);




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
