# [Day 08 : ML 工程師職責與分工](https://ithelp.ithome.com.tw/articles/10260962)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10260962)

資料團隊組建
------

-   當各行業意識數據帶來業務成長新動能時，追求卓越的企業意識到要充分運用企業數據，必須組建專門數據團隊，期待專業團隊具有提煉數據價值的慧眼，也期待落實至實際產品服務。
-   依據 LinkedIn 2020 年針對美國[新興工作報告](https://business.linkedin.com/content/dam/me/business/en-us/talent-solutions/emerging-jobs-report/Emerging_Jobs_Report_U.S._FINAL.pdf)，伴隨著數位轉型、AI 應用、IOT 物聯網及雲端的需求，與 AI 及 Data 相關的「 AI 專家」、「資料科學家」和「資料工程師」，以及具有現代佈署產品在雲端運行相關的「站點可靠工程師 Site Reliability Engineer(SRE)」、「雲端工程師」、「後端工程師」職務，都在的關鍵前 15 名中。
-   另外值得一提的是，近年因 DevOps 與 SRE 工程文化興起，企業意識到 DevOps 無限循環的好處，讓原本開發、維運各為不同團隊明確分工的事情，能彼此溶入循環降低溝通成本， MLOps 需要結合資料工程、資料科學、 SRE /雲端/後端/(前端)工程。

MLOps 及 ML 服務中需要的各種角色
---------------------

-   領域專家
-   資料科學家
-   資料工程師
-   軟體工程師
-   DevOps
-   風控經理/稽核者
-   ML架構師

ML 服務中的主要的角色
------------

### 資料科學家 Data Scientist

-   被喻為21世紀最性感的工作，基於企業對於數據的重視需求持續增長，技術領域可能包含但不限於 Machine Learning, Data Science, Python, R, Apache Spark ，主要工作為利用資料科學技術洞察數據、執行需求訪談、資料收集、資料清理 、模型建置、資料視覺化等任務，並且從事機器學習/深度學習/文字探勘演算法開發應用，具有數據資料解析、數據洞察能力及良好邏輯分析與程式開發能力可勝任。

### 資料工程師 Data Engineer

-   數據已迅速成為每個公司最寶貴的資源，需要精明可以構建基礎設施以保持其井井有條的工程師，企業聘用需求自持續增長。
-   主要工作：關注如何實踐資料基礎設施與工作流程，包含資料 ETL、並依數據基礎工程需求建構 Apache Kafka、Apache Flink、Hadoop、Apache Spark 等工作流程管道，可以想像是大數據有關工程的部分，並且因為公有雲平台的發展， AWS 成為加分或必備技能。

### 機械學習工程師 Machine Learning Engineer(ML Engineer)

-   新興的職務需求，工作重視如何從頭開始設計和構建新的數據管道，一直到將它們部署到生產環境，並且能配合錯誤分析及異常偵測，維持已佈署服務的可用及再現性，觀察目前機械學習工程師職缺有與 AI 工程師混用的情形，但以專業分工來看，偏重佈署及佈署後的維運工作，而且是主責在 ML 模型，具有 MLOps 經驗尤佳，諸如有 Airflow, Bigquery, Feats, MLFlow, Kubeflow 相關經驗，也牽涉自動化 CI/CD 與 CT 的任務。
-   [微軟在 Azure 說明什麼是機器學習](https://azure.microsoft.com/zh-tw/overview/what-is-machine-learning-platform/#uses)說明「機器學習工程師的工作是什麼？」如下:
    
    > -   機器學習工程師會將來自各種資料管線收集的未經處理資料，轉譯成可以視需要套用及調整的資料科學模型。
    > -   機器學習工程師會將該結構化資料連線到與他們共事之資料科學家所定義的模型。
    > -   此外，機器學習工程師也會開發演算法，並建置可讓機器、電腦及機器人處理傳入資料並識別模式的程式。
    

團隊現況與人才供需情形
-----------

-   觀察不同企業數據團隊的期待與任務不一，隨著 ML in Production 越來越成熟，會更重視佈署與營運的狀態，相關人才需求越來越強烈。
-   目前也觀察到學校培育人工智慧領域人才成長趨勢，資料科學雖譽為最性感的職業，也是學子在這波AI浪潮下期待的工作，有僧多粥少的情形，但企業也意識到有資料基礎建設、ETL、SRE 等工程實務需求，可以透過專業分工及軟體開發管理方式合作。
-   在組建資料分析的最小戰鬥單位而言， [Python for DevOps](https://www.books.com.tw/products/0010873248) 一書指出一位資料科學家大約需要3至5位資料工程師及 ML 工程師，故有意踏入數據領域的求職者，加強版本控制、容器化管理、雲端平台應用經驗，並具備 ML 工程實務所需之資料蒐集、資料清理、特徵工程、模型訓練、驗證、除錯優化及佈署營運經驗尤佳， MLOps 是很好的加分項目，學校較難培育 ML 工程師人才，需求卻更大也更適合資訊工程人才發揮，隨著機械學習設計模式成形，期待在需求缺口增加的情況下，業界也能不吝給願意投入 MLOps 任務者工作機會。
-   另鑒於學校在近期主打人工智慧產業應用的科系增加，也希望學校跟在學學子意識到所謂的數據分析與應用，不僅是演算法、各種模型調參的問題，也能接觸 CI/CD 能力學習環境，對未來職涯發展有正面助益。

小結
--

-   MLOps 隨著企業關注用於生產的機械學習趨勢，ML工程師需求將越來越大，透過引述 Linkin 的觀察說明持續增長的趨勢，也希望更多角色能實現 AI 落地的目標。
-   另外微軟在 [Azure 說明什麼是機器學習](https://azure.microsoft.com/zh-tw/overview/what-is-machine-learning-platform/#uses)一文說明機器學習工程師的工作，可以看出是以ML整體系統為出發到佈署營運的工程端對端工作，可見 ML 工程師工作將日益重要。  
    ![/images/emoticon/emoticon12.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon12.gif)

參考
--

-   [2020 Emerging Jobs Report - Linkin](https://business.linkedin.com/content/dam/me/business/en-us/talent-solutions/emerging-jobs-report/Emerging_Jobs_Report_U.S._FINAL.pdf)
-   [Python for DevOps, 2020](https://www.books.com.tw/products/0010873248)
-   [How to build a data analytics dream team - MIT , 2020/5](https://mitsloan.mit.edu/ideas-made-to-matter/how-to-build-a-data-analytics-dream-team)
-   [Azure 什麼是機器學習？](https://azure.microsoft.com/zh-tw/overview/what-is-machine-learning-platform/#uses)
