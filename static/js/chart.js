var chartDom = document.getElementById('Chart');
var myChart = echarts.init(chartDom);
var option;

option = {
  xAxis: {
    type: 'category',
    data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun',"1","2","3","4","5"]
  },
  yAxis: {
    type: 'value'
  },
  series: [
    {
      data: [150, 230, 224, 218, 135, 147, 260, 1, 1, 1, 1, 1],
      type: 'line'
    }
  ]
};

option && myChart.setOption(option);
