# [Day 03 : ML in Production 的挑戰](https://ithelp.ithome.com.tw/articles/10259314)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10259314)

在 [Day2](https://ithelp.ithome.com.tw/articles/10258861) 提到什麼是用於生產的機械學習 ML in Production ，今天來談用於生產的機械學習所遇到的挑戰，主要挑戰包含:

-   要是整合性的機械學習系統。
-   要持續地在生產環境中維運。
-   要處理持續改變的資料流。
-   要控制電腦運算資源與成本。

而之所以成為挑戰，筆者設想學用兩端落差，摘述如下:

"O 到 1" 與 "1 到 N" 是不同層次的問題
--------------------------

-   相信學習人工智慧領域的人有越來越早觸及的趨勢，您可能在大學就有優秀又有熱誠的老師開設 "從0到1" 的人工智慧、機械學習、深度學習課程，研究所、研究室也有專題或論文指引方向解決問題，甚至您可能苦笑又自豪地說我就是個自學仔，取之於網路資源開始 AI 之路，您的堅持與努力都讓筆者相當欽佩，也真心崇拜與感謝如：李弘毅、林軒田、蔡炎龍等大神，猶如醍醐灌頂苦海明燈。
-   但 "1到N" 指的是將已有的 AI 模型佈署到服務數以萬計、億計人流的商業服務時，現況是學子們非常難有產業經驗，甚至許多企業也摸索不得其門，畢竟購置算力開銷不小，投入成本跟效益難以估算。也有來自 [KDnuggets 的文章](https://www.kdnuggets.com/2021/07/mlops-best-practices.html)引述 2019 年有 87% AI 止步於落地之前，另外一篇文章也指出 AI 新創陣亡率高達 9 成。

學習 ML/DL 像挑戰登高，回首還有 Data 洪流
---------------------------

-   AI 演算法興起討論自 2016 年 AlphaGo 進展從 DNN、CNN、RNN、LSTM 到 GAN、BERT、Tramsformer、GTP-3、YOLO 等屢有突破，有幸能躬逢其盛，要追的論文很多，就像高山一樣追不完，能看懂演算模型誠屬不易、歷經層層神經網套用 PyTroch 、 TensorFlow 的摧殘，在學習之路上回頭山峰很高，驚覺實踐時還有個資料洪流，時不時還要溯溪泛舟。
-   資料坑是在挽起袖子實際訓練 AI 時就會遇到的難題，蒐集資料、標註資料、清洗資料及特徵工程，或許在初學時還可以先拿 IRIS、MNIST、波士頓房價、 ImageNet 等資料集「硬 Train 一波」再說，但是商務情境之下資料品質的問題會被放大，諸如手寫辨識的目標是連人都看不懂鬼畫符，不知何時"割韭菜"跟"航海王"詞義已經變化，AI 要落地就不得不面耗費80%心力與資料搏鬥，「屢 Train 不停」。

AI 數據競賽練功坊，與用於生產的任務有差
---------------------

-   學習人工智慧領域，透過AI建模解決問題，或提出新型模型應用、參加 Kaggle 數據競賽，是現今主流學習人工智慧的趨勢，也是培養 AI 人才的好方法，但仍有盲點存在:
    1.  競賽的資料可控程度較高，是「給定的資料集」，並且有特徵欄位說明，簡化了其變動程度，好讓競賽焦點集中在資料標註、特徵工程、模型優化，提出更好的預測結果。
    2.  追求預測的準確率 Accuracy 指的是平均準確率，最終比賽目標關注整體準確率，但對於隱藏在資料集的極端分佈或錯誤修正並非關注焦點，「 Responsible AI 」討論到 AI 系統的公平性、可解釋性、隱私性與安全性， ML 產品服務經不起種族、性別、就業歧視的指責，但數據競賽不見得要為此考量。
    3.  提交一次性預測模型給競賽平台或給定研究解決方案，優異訓練成果，測試也不錯，但佈署到現實世界所需的「持續優化任務」，通常到了業界才會碰觸。
        
        > 吳恩達也舉辦了[以資料為中心的ML競賽](https://https-deeplearning-ai.github.io/data-centric-comp/)，改鎖定ML模型，參賽者以資料工程手段改進訓練成果，就是希望能把焦點轉移到改進資料品質中。
        

不願意面對的真相，你的 ML Code 只是冰山一角
--------------------------

-   引述 [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) 論文，除了你的 ML CODE ，真實情境下要考量的很多，只有裡面黑黑的一小塊叫做 ML Code 。
-   ![](https://i.imgur.com/eBzHkvF.png)

營運燒錢難以想像
--------

-   摘述 [DeepMind 巨虧 180 億、加拿大獨角獸遭 3 折賤賣，AI 公司為何難有「好下場」？](https://buzzorange.com/techorange/2021/01/07/deepmind-deficit-ai-company-startup-survival-challenge/)一文指出，以 Open AI 著名的文本生成算法 GPT 為例，一個擁有 15 億參數的模型，每小時訓練都要花費 2048 美元，而類似於 DeepMind 的 AlphaGo 算法，成功之前需要至少完成數百萬次自我博弈，光訓練費就要花 3500 萬美元。

最後，我們學習過程喜愛的筆記本環境，只是工具鏈的一環
--------------------------

-   Jupyter Notebooks 是數據科學團隊善用的工具，學習與分享都很方便，但部署 ML 到生產情境時，所要接觸的絕不能僅止於此，以學習經驗而言，除非刻意接觸或有演練機會，不然同時具備 ML + DevOps 能力者是相當缺乏的，也是非戰之罪。

是說有沒有 ML 完整解決方案?
----------------

-   雲端大廠有提供 ML 完整解決方案，舉例如 [Azure Machine Learning Studio](https://docs.microsoft.com/zh-tw/dynamics365/customer-insights/audience-insights/machine-learning-studio-experiments) 就是，個人覺得 Azure 最好的地方就是文件跟說明完整，入門起來較不排斥，可以在您熟悉的資料科學領域加值，本系列文後續再作介紹。
-   ![](https://docs.microsoft.com/zh-tw/dynamics365/customer-insights/audience-insights/media/azure-machine-learning-studio-experiment.png)
    -   圖片來源: [Azure Machine Learning Studio](https://docs.microsoft.com/zh-tw/dynamics365/customer-insights/audience-insights/machine-learning-studio-experiments)

小結
--

-   今天整理了「用於生產中的機械學習」與學習取向、研究取向、模型競賽取向的 ML/DL 領域2者間的差異，意識到中間的挑戰與落差原因，如果您正在起步學習人工智慧，那本篇可以提供您不同面向的思考觀點。
-   希望後續能持續提供有助益的乾貨，燒腦中，我們明天見。  
    ![/images/emoticon/emoticon07.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon07.gif)

參考
--

-   [Machine Learning Engineering for Production (MLOps) 系列課程](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)
-   [KDnuggets的文章](https://www.kdnuggets.com/2021/07/mlops-best-practices.html)
-   [Hidden Technical Debt in Machine Learning Systems, 2015](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)
-   [DeepMind 巨虧 180 億、加拿大獨角獸遭 3 折賤賣，AI公司為何難有「好下場」？](https://buzzorange.com/techorange/2021/01/07/deepmind-deficit-ai-company-startup-survival-challenge/)
-   [以資料為中心的ML競賽](https://https-deeplearning-ai.github.io/data-centric-comp/)
-   [Azure Machine Learning Studio](https://docs.microsoft.com/zh-tw/dynamics365/customer-insights/audience-insights/machine-learning-studio-experiments)
