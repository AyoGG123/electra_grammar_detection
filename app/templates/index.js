var datas = [];
function f_submit_function() {
清空();

var sentencesValue = document.getElementById("chunks").value;
var chineseValue = document.getElementById("chinese").value;
jsonObj = [];
item = {};
item["chunks"] = [sentencesValue];
item["chinese"] = chineseValue;

//載入中文字...
document.getElementById("state").textContent = "...loading...";

$.ajax({
  //url: "https://127.0.0.1:80/predict", //從這裡更改PORT口 預設是80 本地使用
  url: "https://correct-dogfish-remarkably.ngrok-free.app/predict", //從這裡更改PORT口 預設是80
  data: JSON.stringify({
    chunks: [sentencesValue],
    chinese: chineseValue,
  }),
  type: "POST",
  dataType: "json",
  contentType: "application/json;charset=utf-8",

  success: function (returnData) {
    console.log(returnData);
    //document.getElementById("result_single").innerHTML = "pred ：" + JSON.stringify(returnData.return[0]);

    for (var i = 0; i < returnData.return.length; i++) {
      datas.push(
        JSON.stringify(returnData.return[i]).replace(/['"\[\]]/g, "")
      );
    }
    //document.getElementById("ID").innerHTML = JSON.stringify(returnData.return[0]).replace(/['"]+/g, '');
    //document.getElementById("sentence").innerHTML = JSON.stringify(returnData.return[1]).replace(/['"]+/g, '');
    //document.getElementById("predict").innerHTML = JSON.stringify(returnData.return[2]).replace(/['"]+/g, '');
    document.getElementById("state").textContent = "完成";
    //document.getElementById("test").innerHTML = datas;
    顯示表格();
  },
  error: function (xhr, ajaxOptions, thrownError) {
    console.log(xhr.status);
    console.log(thrownError);
  },
});

jsonObj.push(item); //將item傳入jsonObj裡面
//document.getElementById("input_json").innerHTML = JSON.stringify(jsonObj);
}
function 顯示表格() {
var tbody = document.querySelector("tbody");

// 清空现有的表格内容
tbody.textContent = "";
datas.forEach((elem, index) => {
  var tr = document.createElement("tr");
  tbody.appendChild(tr);
  //var count = 0
  temp = elem.split(",");
  temp.forEach((elem1, index1) => {
    elem1 = 截斷(elem1);

    console.log("-----------------------");
    console.log(elem1.length);

    var td = document.createElement("td");
    //把对象里面的属性只给td
    td.textContent = elem1;
    tr.appendChild(td);
    console.log(elem1);
  });
});
}
function 清空() {
//document.getElementById("test").textContent = "";
datas = [];
}
function 截斷(n) {
var maxContentLength;

if (window.innerWidth <= 768) {
  // 根据屏幕宽度判断设备类型
  maxContentLength = 15; // 手机设备
} else {
  maxContentLength = 35; // 电脑设备
}

if (n.length > maxContentLength) {
  n = n.substring(0, maxContentLength) + "...";
}
return n;
}