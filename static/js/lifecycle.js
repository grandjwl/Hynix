$(document).ready(function () {
    var table = $('#table_id').DataTable();
    
    datatableEdit({
        dataTable: table,
        columnDefs: [
            {
                targets: 3
            }
        ]
    });

    // 페이지 번호 클릭 시 스타일 변경
    $(".paginate_button").click(function () {
        $(".paginate_button").removeClass("active-page");
        $(this).addClass("active-page");
    });

    // 데이터 테이블 페이지 변경 시 스타일 변경
    table.on("page.dt", function () {
        $(".paginate_button").removeClass("active-page");
        var activePageButton = $(".paginate_button").eq(table.page.info().page);
        activePageButton.addClass("active-page");
    });

    // 초기 페이지에서 현재 페이지 버튼에 스타일 적용
    var initialPageButton = $(".paginate_button").eq(table.page.info().page);
    initialPageButton.addClass("active-page");
});
