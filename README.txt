使用Electra作文法檢測
使用hfl/chinese-electra-180g-base-discriminator預訓練模型後微調
資料使用nlpcc2023公開資料，加上資料處裡

網址：POST http://伺服器ip位置/predict

data type: json

chunk delimiter: 。？！；（皆為全形）
單一chunk最大長度為512個字符
輸入格式：
	{
		"chunks":["chunk1", "chunk2", ...],
		"chinese":"simplified"

	}
	ex.
	{
		"chunks":["东京是日本经济的中心，有很多日本公司以及外国企业。","此外，也是日本购物中心。"],
		"chinese":"simplified"

	}


pred會輸出chunk裡每一個sentence是否有錯誤（sentence delimiter: ，。？！；）
0為無錯誤，1為有錯誤

輸出格式中"chunks"為原始輸入句
輸出格式：
	{
    "return": [
        {
            "id": [
                "1",
                "2"
            ],
            "pred": [
                "[0, 0]",
                "[0, 1]"
            ],
            "chunks": [
                "东京是日本经济的中心，有很多日本公司以及外国企业。",
                "此外，也是日本购物中心。"
            ]
        }
    ]
}