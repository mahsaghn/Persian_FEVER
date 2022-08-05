$(document).ready(function () {
    $('#maintab a').click(function (e) {
        e.preventDefault()
        $(this).tab('show')
    })
});

function ExampleClick(item){
    $("#claim").val($(item).context.innerText)
}