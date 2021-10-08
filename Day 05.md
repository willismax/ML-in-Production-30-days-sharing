# [Day 05 : ML 專案生命週期](https://ithelp.ithome.com.tw/articles/10259989)


###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10259989)


-   從無到有開發 ML 專案到佈署需要 6 至 12 個月不等，在尚未有具體產出的過程中，會有對內部及外部說明進展的機會，能有架構、系統的與合作對象說明是很重要的。吳恩達在 2021 年 4 月在 吳恩達在 2021 年 4 月在 [DeepLearning.AI 發布的電子報](https://read.deeplearning.ai/the-batch/issue-87/)歸納了出 4 大階段，用以描述 ML 專案生命週期，也是後續開發新課程架構的主要輪廓。
    
-   隨後， 2021 下半年在 Coursera 推出上述 [Machine Learning Engineering for Production  
    (MLOps) Specialization](https://www.deeplearning.ai/program/machine-learning-engineering-for-production-mlops/) (MLEP) 系列課程計 4 門，架構如下:  
    ![](https://i.imgur.com/Mc8Cwcp.png)
    
-   您可以看出在較新的ML產品生命週期圖示，整合為四大階段並歸納 7 個主題，用以描述用於生產的機械學習工作流程，這樣的工作流程實際上並非是轉圈圈的循環圖，而是有向無環圖（[Directed Acyclic Graph (DAG)](https://zh.wikipedia.org/wiki/%E6%9C%89%E5%90%91%E6%97%A0%E7%8E%AF%E5%9B%BE)），箭頭表示了工作流程及相依性。
    

架構摘述
----

-   簡述4個階段與7大主題，之後文章會以此架構再進一步說明:
    
    -   範疇Scoping
        -   定義專案: ML專案的商業考量目標。
    -   資料Data
        -   定義資料與建立基準。
        -   標註與組織資料。
    -   建立模型Modeling
        -   選擇與訓練模型。
        -   錯誤分析。
    -   佈署Deployment
        -   在生產情境中部署。
        -   監控與維運系統。
-   上圖點出用於產品的ML專案需要注意的事情，舉例如：
    
    -   標註數據：
        -   您很難一次性的取得乾淨的資料，可以先標註數據後，透過建立指引檢查並改具數據，(本系列文後續會說明作法)。
    -   訓練模型：
        -   建構AI系統的輸入是要決定使用哪些「數據」、「超參數」和「模型架構」。與其過度考慮這些選擇，不如訓練初始模型，然後透過錯誤分析推動改進。
    -   部署和監控：
        -   在部署ML系統時，您需要設計符合需求的監控指標與儀表板，以嘗試發現概念漂移或資料漂移。

Azure 的 ML 專案工作流程
-----------------

-   通常會在具有目標和目標的專案中開發模型。 專案通常牽涉到一個以上的人。 使用資料、演算法和模型進行實驗時，會反復開發。
    
    -   ![](https://i.imgur.com/RoSkuFL.png)
        -   圖片來源: [Azure](https://docs.microsoft.com/zh-tw/azure/machine-learning/overview-what-is-azure-machine-learning)
-   [Azure Machine Learning Pipeline](https://docs.microsoft.com/zh-tw/azure/machine-learning/concept-ml-pipelines) 可以包含 ML 生命週期的相關工作，依該文件說明包含如下，並且有設計工具協助:
    
    -   資料準備，包括匯入、驗證和清除、改寫和轉換、正規化以及暫存。
    -   訓練組態，包括參數化引數、檔案路徑，以及記錄/報告組態。
    -   有效且重複地訓練和驗證。 效率可能來自指定特定的資料子集、不同的硬體計算資源、分散式處理和進度監視。
    -   部署，包括版本控制、調整、佈建和存取控制。
    -   ![](https://docs.microsoft.com/zh-tw/azure/machine-learning/media/concept-designer/designer-drag-and-drop.gif)
        -   資料來源: [Azure Machine Learning 設計工具](https://docs.microsoft.com/zh-tw/azure/machine-learning/concept-ml-pipelines#building-pipelines-with-the-designer)

小結
--

-   您在搜尋網路上諸多傳統的 ML pipeline 流程圖，會發現工作流程步驟相當複雜且一致，這也是資料科學、資料工程、機械工程、佈署維運彼此之間的專業價值。
-   本系列文引用的 ML 專案生命週期相當清晰，也可以更能聚焦在「資料為中心」的任務流程與工作價值，希望對您有幫助。

參考
--

-   [DeepLearning.AI The Batch, 2021/4/15](https://read.deeplearning.ai/the-batch/issue-87/)
-   [Lifecycle of an ML Project](https://read.deeplearning.ai/the-batch/iteration-in-ai-development/)
-   [Machine Learning Engineering for Production  
    (MLOps) Specialization](https://www.deeplearning.ai/program/machine-learning-engineering-for-production-mlops/)
-   [Directed Acyclic Graph](https://zh.wikipedia.org/wiki/%E6%9C%89%E5%90%91%E6%97%A0%E7%8E%AF%E5%9B%BE)
-   [Azure Machine Learning 管線](https://docs.microsoft.com/zh-tw/azure/machine-learning/concept-ml-pipelines)
