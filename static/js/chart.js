var chartDom = document.getElementById('Chart');
var myChart = echarts.init(chartDom);
var option;

var initialMseData = [
    { date: '2023-08-01', delta: 3 },
    { date: '2023-08-02', delta: 2.3 },
    { date: '2023-08-03', delta: 1.5 },
    { date: '2023-08-04', delta: 1.2 },
    { date: '2023-08-05', delta: 0.9 },
    { date: '2023-08-06', delta: 1.5 },
    { date: '2023-08-07', delta: 2.6},
    { date: '2023-08-08', delta: 2.1 },
    { date: '2023-08-09', delta: 3.1 },
    { date: '2023-08-10', delta: 4.6 },
    { date: '2023-08-11', delta: 5.9 },
    { date: '2023-08-12', delta: 3.6 }
];




var option = {
  xAxis: {
    type: 'category',
    data: initialMseData.map(item => item.date),
    name: 'Date', // x축 이름 설정
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
    name: 'delta',
    nameTextStyle: {
      color: 'black',
      fontSize: 20,
      fontWeight: 'bold',
    },
    axisLabel: {
      show: true,
      textStyle: {
        fontWeight: 'bold',
        color: 'black',
        fontSize: 15, // 텍스트 크기 조정
      },
    },
  },
  
  

  series: [
    {
      data: initialMseData.map(item => item.delta),
      type: 'line',
      lineStyle: { // 기본 그래프 선 스타일
        color: 'blue', // 파란색으로 변경
      },
      markLine: {
        data: [
          {
            yAxis: 5,
            label: {
              show: true,
              fontWeight: 'bold',
              color: 'red',
              formatter: 'Threshold' // 여기에 표시할 내용을 넣어주세요
            }
          }
        ],
        lineStyle: {
          color: 'red',
        },
      },
    },
  ],
};

myChart.setOption(option);
