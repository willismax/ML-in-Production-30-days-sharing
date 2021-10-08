# [Day 04 : 以資料為中心的人工智慧 Data - Centric AI](https://ithelp.ithome.com.tw/articles/10259708)


###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10259708)


-   垃圾進垃圾出「 Garbage in, garbage out 」，不去檢視垃圾有多垃圾的情況下，用再好的模型都是垃圾!![/images/emoticon/emoticon40.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon40.gif)
-   [Day 03](https://ithelp.ithome.com.tw/articles/10259314) 有提到 AI 數據競賽用於生產的任務的差別，吳恩達也舉辦了[以資料為中心的 ML 競賽](https://https-deeplearning-ai.github.io/data-centric-comp/)，改鎖定 ML 模型，參賽者以資料工程手段改進資料品質以增進訓練成果，與主流 Kaggle 數據競賽改 Model 調參不同，就是希望能把傳統「以模型為中心」的焦點目光轉移到更系統化的改進資料品質。今天來談以數據為中心的人工智慧 Data-Centric AI 的思辨:  
    ![](https://i.imgur.com/tjYqZlb.png)

> 圖片修改自 [Data-centric AI: Real World Approaches  
> ](https://www.youtube.com/watch?v=Yqj7Kyjznh4)

資料為中心與模型為中心的焦點比較
----------------

-   模型為中心的 AI（Model-centric AI）
    -   使用擁有或給定的資料集，使模型的效果越佳越好，傳統的作法。
    -   固定資料，持續提升 Algorithm/Model 最佳解。
-   資料為中心的 AI（Data-centric AI）
    -   資料品質是參數，持續改進資料品質，並且允許複數模型的工作流程，這樣的過程是反覆的，而且是喔有系統的進行。
    -   固定 Code ，持續提升資料品質。

資料為中心的 AI 可以做到的事情
-----------------

-   針對特定子資料集/切片進行資料優化。
-   針對持續蒐集到的新資料進行錯誤分析及改進資料。
-   在錯誤分析與解釋 AI 甚至到改進局部預測能力特別有用。

資料為中心的 AI 工作流程持續循環
------------------

-   改進資料不是一次性的任務，而是持續循環的過程，資料為中心的 AI 工作流程為訓練資料、錯誤分析以決策、改進資料等 3 項任務持續循環。在此引用 [DeepLearning.AI 發行的電子報](https://read.deeplearning.ai/the-batch/issue-105/)圖片:
    -   ![](https://i.imgur.com/mCX3Ytm.png)
        -   圖片來源: [Deeplearning.ai: the batch](https://read.deeplearning.ai/the-batch/issue-105/)

一些實務的改進資料作法
-----------

-   在吳恩達 Deeplearning.ai 團隊在 [Data-centric AI: Real World Approaches  
    ](https://www.youtube.com/watch?v=Yqj7Kyjznh4)直播中，提及了些有趣可以改善訓練成果的 6 個做法，在非結構資料(圖片、文字等)、資料量較小的情況下，改進資料品質相當有幫助:
    -   方法1: 將連續性的標籤Ｙ呈現一致
        
        -   `X->y` 如果是呈現隨機對應，譬如藥丸瑕疵檢測、手機刮痕瑕疵檢測，以刮痕長短為 x，瑕疵與否為 y ，本來預期刮痕超過一定長度會被判定為瑕疵，但因為是人為標註的結果，而且標註來自不同人、不同判定標準，判斷標準不一將影響訓練成果。
        -   此時可以將圖片透過刮痕長度 x 排序，並「決定」瑕疵 y 的判斷基準，經過梳理後會呈現邏輯回歸的分布狀態，比原來飄忽繁亂的標註改善許多，如下圖刮痕 2mm 的判讀可以修正。
        -   ![](https://i.imgur.com/KYw6HFs.png)
    -   方法2: 讓非連續性的標籤一致。
        
        -   檢查同個意義的標籤應要一致，像是 "people" 與 "human" 混用，另外對於性別盡量採用中性詞彙。
        -   使同個標註標的出現時，如2個以上的刮痕，標註數量一致、邊界大小一致。
    -   方法3: 出現的未知標籤，定義他並且寫入指引。
        
        -   指引應包含說明、舉例、標註範例、讓人疑惑地類似狀況舉例。
    -   方法4: 取捨模糊資料，資料越多不見得越好。
        
        -   非結構的模糊不清的資料，如果連人類專家都無法明確判斷，捨棄該筆資料也會改善訓練結果，但如果必須要預測模糊資料，則應設法有判斷指引。
    -   方法5: 專注分析並改善有瑕疵的子資料集。
        
        -   改善有誤差的子資料集，可以讓整體預測準確率上升。

小結
--

-   對於回頭檢視資料是用於生產的機械學習必須任務，實務上因為資料偏移、概念偏移造成的模型預測準確率下滑，可以設計觸法機制做自動化重新訓練。
-   另外，對於資料偏斜的問題，可以細細檢視資料及，並且修改標註與預測結果，提供更中性、公平而非偏見的判斷與輸出。
-   吳恩達倡議"以資料為中心的 AI "，揭示2021年機械學習更關注部署營運階段，反思面對資料的必要性。有幸能一同見證逐漸落地的情境，我們下篇見。

參考
--

-   [以資料為中心的ML競賽](https://https-deeplearning-ai.github.io/data-centric-comp/)
-   [Data-centric AI: Real World Approaches  
    ](https://www.youtube.com/watch?v=Yqj7Kyjznh4)
-   [Deeplearning.ai: the batch - 2021/8/19](https://read.deeplearning.ai/the-batch/issue-105/)
