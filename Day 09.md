# [Day 09 : 用於生產的機械學習 - 定義範疇 Scope](https://ithelp.ithome.com.tw/articles/10261352)


###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10261352)


-   在 [Day 05 ML 專案生命週期](https://ithelp.ithome.com.tw/articles/10259989)介紹分為 4 個階段與 7 大主題，第 1 個階段為「定義範疇 Scoping」，相較其他 3 個階段，Scoping 較偏管理知識面的討論，有專案發想經驗的您應該不陌生，我們會試著把商業命題轉換為 AI 命題。

![](https://i.imgur.com/vX4d9EP.png)

以下延續吳恩達 [Machine Learning Engineering for Production  
(MLOps) Specialization](https://www.deeplearning.ai/program/machine-learning-engineering-for-production-mlops/) 系列課程所提的 ML 專案生命週期架構進行說明。

定義商業命題
------

-   一個明確的商業命題可以引領專案往好的方向發展。對參與其中的您而言，專案方案理解越清晰，團隊運行及專案發展也越容易成功。
-   在命題的過程中，要思考的包含:
    -   專案的目的是什麼?
    -   要如何衡量績效?
    -   需要哪些資源?

### 專案的目標是什麼? 透過腦力激盪吧!

1.  設想目標有時很單純(老闆說要做)，也有採取一種比較開放的方式，透過「對商業問題腦力激盪」可能有較佳的設想，畢竟 ML 專案需要企業資源投入，在起步階段，也面臨較多的不確定性。
2.  簡單的想法是羅列出面臨的「問題」，激盪對應的「解決方案」，排列問題的重要性以及時效性。
3.  同時，再進一步聚焦這個問題一定要 AI 解決嗎? 如何解決? 「對 AI 解決提案的腦力激盪」，不是所有命題都需要藉由 AI 完成的，開放且審慎的評估適切的目標達成做法，總比頭洗下去卻方向錯誤的好。

-   設定專案目標腦力激盪的過程中，也同時「評估可行性與潛在價值」，好的開始是成功的一半，毋須躁進。
-   舉例一些對於 AI 相關的命題，例如:
    -   改進建議系統。
    -   改進搜尋結果。
    -   改進分類系統。
    -   產品定價優化...

### 要如何衡量績效、定義里程碑與資源?

-   一旦作為 ML 專案，績效衡量可以用不同的指標綜合評判(順便做個儀表板？)
    -   基於模型的準確率、loss、RMSE、F1等。
    -   系統效能的延遲、計算時間、耗能等。
    -   商業成果，如投報率等。
-   衡量績效的標準，可以參考:
    -   既有服務展現的水準。
    -   開源專案展現的水準。
    -   基於「人類表現的水準」。非結構性資料(圖片、聲音、文字)可以用人類表現水準 HLP (Human-level Performance) 來設定目標。 HLP 一詞來自吳恩達 deeplearning.ai 提出，在其[電子報](https://blog.deeplearning.ai/blog/the-batch-ai-predicts-the-vote-face-recognition-looks-for-criminals-model-cow-makes-milk-transformers-prove-theorems)中討論基於 HLP 而非追求更高績效的想法。
    -   有限的時間基準。
    -   在較不明確的狀況下，可以透過標竿學習、概念驗證 POC 掌握專案輪廓。
-   需要哪些資源?
    -   啟動專案的資源規劃，如所需人力、時間、成本、設備等。

### AI模型的績效一定要超過人類水準?

-   在衡量模型績效時可能會落入一個迷思，就是AI模型的績效一定要超過人類水準，但如果分為結構性資料、非結構性資料(圖片、聲音、文字)衡量績效，確實結構性資料的績效可能大於 HLP ，但非結構資料如果連人類都無法判讀，又如何檢視您的 ML 模型表現是否真的如此優異?
-   舉例來說瑕疵檢測、語音辨識，在遇到逼近或略為大於 HLP ，您可以將目光移至有關安全性、偏見、稀有類別的性能。

| \ | 非結構資料 | 結構資料 |
| --- | --- | --- |
| 新服務 | HLP | Benchmark |
| 既有服務 | 過往服務水準、HLP | 過往服務水準、Benchmark |

### 確認沒有道德疑慮

-   企業服務是需要負社會責任的，在進行 ML 專案開頭時，應該確認專案有無道德疑慮，是否有潛在的偏見風險。您不會希望辛苦做出來的人臉辨識，卻發生像[臉書AI將黑人影片誤標為靈長類](https://www.ithome.com.tw/news/146548)的狀況。

小結
--

-   本日簡要介紹在定義範疇階段可以的流程及注意的事情，可以透過腦力激盪探詢欲定義問題到構思解決方案，進一步釐清是否要發起 AI 專案，衡量績效的方式與績效基準，以及需要的資源。
-   人類的表現 HLP 在聲音、影片、文字的辨識能力是很強大的，讓您的 ML 能達到 HLP 就有替代人力的機會，或許您可以轉移注意力到預測伴隨的安全與道德問題。

參考
--

-   [Machine Learning Engineering for Production  
    (MLOps) Specialization](https://www.deeplearning.ai/program/machine-learning-engineering-for-production-mlops/)
-   [Deeplearning.ai the batch - 2020/11/19](https://blog.deeplearning.ai/blog/the-batch-ai-predicts-the-vote-face-recognition-looks-for-criminals-model-cow-makes-milk-transformers-prove-theorems)
-   [臉書AI將黑人影片誤標為靈長類](https://www.ithome.com.tw/news/146548)
