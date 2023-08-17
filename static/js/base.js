$(function() {
  // 보이기 | 숨기기
  $(window).scroll(function() {
    //  if ($(this).scrollTop() > 50) { //250 넘으면 버튼이 보여짐니다.
    //        $('#topBtn').fadeIn();
    //        } else {
    //        $('#topBtn').fadeOut();
    //  }
    if( $(this).scrollTop() > 50 ){
      $("#wrap").addClass("on");
    }
    else{
      $("#wrap").removeClass("on");
    }
  });

  // 버튼 클릭시
  // $("#topBtn").click(function() {   
  // $('html, body').animate({
  //   scrollTop : 0    // 0 까지 animation 이동합니다.
  //  }, 800);          // 속도 400
  //  return false;
  //  });
  $("#wrap").click(function(){
    window.scrollTo({top : 0, behavior: 'smooth'}); 
    });
 });
 
//  $(window).scroll(function(){
    
//   if( $(this).scrollTop() > 100 ){
//     $("#top_btn").addClass("on");
//   }
//   else{
//     $("#top_btn").removeClass("on");
//   }
  
// });
  
// $("#top_btn").click(function(){
// window.scrollTo({top : 0, behavior: 'smooth'}); 
// });