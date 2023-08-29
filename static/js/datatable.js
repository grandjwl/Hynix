$(document).ready(function () {
    var table = $('#table_id').DataTable();
    
    datatableEdit({
        dataTable: table,
        columnDefs: [
            {
                targets: 2

            }
        ]
    });
});
