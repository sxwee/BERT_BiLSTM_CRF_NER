
function OpenLocalFile(){
    document.getElementById('openfile').click()
}

function saveFile(){
    $.ajax({
            type: "GET",
            url: "/ner/save/",    //后台处理函数的url
            success: function (data) {  //获取后台处理后传过来的result 
                alert("命名实体集返回成功");
                window.location.href = "/ner/save/?file='temp.txt'"
            },
    });
}

function changeMethod(e) {
    var file = e.files[0]
    var reader = new FileReader();//新建一个FileReader
    reader.readAsText(file, "UTF-8");//读取文件 
    reader.onload = function(evt){ //读取完文件之后会回来这里
        var fileString = evt.target.result; // 读取文件内容
        //$("#content").val() = fileString;
        document.getElementById("content").innerText = fileString
    }
}

function startRecognize(){
    var fileString = document.getElementById("content").innerText
    if(fileString.length == 0){
        alert("识别内容为空，请导入文档或手动输入")
    }
    else{
        $.ajax({
            type: "POST",
            url: "/ner/recognize/",    //后台处理函数的url
            data: {
                "fileString":fileString
            },
            success: function (result) {  //获取后台处理后传过来的result
                alert("识别完成") 
                result = JSON.parse(result)
                entity_nums = JSON.parse(result['entity_nums'])
                entity_labels = JSON.parse(result['entity_labels'])
                fileString = result['fileString']
                refreshData(entity_nums,entity_labels)
                document.getElementById("content").innerHTML = fileString
            },
        });
    }
}

function refreshData(nums,labels){
    //alert(labels)
    //刷新数据
    let option = myChart.getOption();
    option.xAxis[0].data = labels;
    option.series[0].data = nums;
    myChart.clear(); 
    option & myChart.setOption(option);
}


