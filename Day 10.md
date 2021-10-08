# [Day 10 : 用於生產的機械學習 - Data Define 與建立基準](https://ithelp.ithome.com.tw/articles/10261664)


###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10261664)

-   接續介紹 ML 專案生命週期，本日說明第 2 階段「資料 Data」的工作流程，依其說法分為2大步驟，分別為「定義資料及建立基準」及「標註及特徵工程」。是4個階段中相當重要的流程，由於您已上線的 ML 系統是會持續不斷的資訊流，也是傳統以模型為中心的 AI 較為棘手的議題，故分為兩個步驟介紹，本日介紹專注在定義資料與建立基準。

定義資料
----

-   簡單的說，就是定義資料的 X 與 y。
-   此階段試圖將「商務需求/使用者服務的的需求」轉化爲特徵 X 與標籤 y。
-   譬如提供電商購物 App 服務：
    -   使用者有接收促銷訊息、限時完成折扣品購物提醒需求。
    -   商業上有增加黏著度、轉換率與促銷提醒。
    -   X: 使用者消費紀錄、時間戳記、明細分類等。
    -   y: 促銷訊息點擊率、用戶評價等。

### 定義資料的工程思考

-   資料標籤是否具有連續性，牽涉到標註資料是否有也呈現連續性、X推導y可以屬於回歸/分類問題。
-   蒐集資料的頻率，屬於串流或批次。
-   蒐集資料前、後的屬性設計，譬如通勤紀錄了經緯度地理位置，特徵工程時需要轉為距離的狀況。
-   資料的安全與隱私考量，也關注著資料的公平性，解釋力。
    -   舉例如:提示儲存隱私資料的風險、提供使用者控制提供何種資料的權限。
    -   確保少數集合能受到公平的結果，譬如不該提供素食主義者或特定宗教飲食忌諱的餐廳。

蒐集資料
----

-   依據定義資料進行蒐集，蒐集來源包含開放資料、使用者數據、網路爬蟲、服務商 API 等。
-   垃圾進，垃圾出，蒐集資料時做好資料 ETL。
-   資料蒐集與使用，要特別注意「資料安全、用戶隱私」，以及在提供「公平」的服務，「消逆偏見」與歧視。

### 關注總是在變的資料\- Drift & Skew

由於資料隨著時間的推移變化，新概念還來不及定義，未分類的 "Unknow" 資料越來越多，而且舊概念的意涵轉變，會減損模型預測的準確程度。

#### 飄移 Drift

對於用於生產的機械學習，遇到的飄移情形可以歸納為資料飄移與概念飄移兩種:

-   資料飄移 Data Drift
    -   輸入的數據 x 本身的改變。
    -   原本 ML 系統在舊數據效果良好，伴隨著未知的數據出現，在未知數據區域表現效果差，ML 系統預測效果衰退。
-   概念飄移 Concept Drift
    -   x 與 y 關係改變。
    -   當目標變量本身的統計屬性發生變化時，就會發生概念漂移。世界已經改變，模型需要更新。
    -   可以是漸進的、突發的、反覆/季節性的。

#### 偏斜 Skew

-   指兩個不同版本的資料，比較之下發生了 Schema 、特徵 x 及資料分布偏斜的差異。
-   版本可能是訓練資料與現實資料的差異，前次已佈署的訓練結果與本次訓練結果之間的差異。
-   需要能比對資料版本之間差異的工具協助驗證。

### 持續不斷的改進資料品質

-   用於生產的機械學習常會回頭檢視並改進資料的品質，是持續循環的過程，在[Day 04](https://ithelp.ithome.com.tw/articles/10259708) 以數據為中心的 AI 引用 [Deeplearning.ai](https://read.deeplearning.ai/the-batch/issue-105/) 的循環圖，訓練模型、錯誤分析以決策、改進資料等 3 項任務，在 ML 生命週期中持續循環。
-   您會需要持續關注資料的主因是用於生產的機械學習，接收到的資料是持續流動的。

建立基準
----

-   根據 Scope 所定義的專案目標，設定欲達成的基準值或警戒值。
-   基準可來自過去的經驗、領域知識、先前績效、參考標竿的績效、 HLP 等。
-   如果能針對不同干擾情境(例如:語音辨識時的交通噪音、工廠內設備噪音)建立不同基準，對於後續不同情境改善與監控會有幫助。

驗證資料
----

-   視覺化的驗證資料集，比對現有與之前的情況是否有偏斜與飄移，需要工具協助。
-   [TFDV](https://blog.tensorflow.org/2018/09/introducing-tensorflow-data-validation.html) 是由 Google 開源，可以檢視資料健康情形的模組，作為 TFX 端對端 (end to end) 流程組合的一部份組件，也可以單獨使用，並且與 Notebook 環境整合。詳細介紹將獨立一篇說明。
    -   ![](https://1.bp.blogspot.com/-vKmxmgPWgv4/XgUuCIHmGhI/AAAAAAAACEw/NC18jPUUpZUjEU6N39JcfTwd5C49tZoKgCLcBGAsYHQ/s1600/figure2.gif)
        -   圖片引用自 [Introducing TensorFlow Data Validation: Data Understanding, Validation, and Monitoring At Scale](https://blog.tensorflow.org/2018/09/introducing-tensorflow-data-validation.html)
-   在微軟 Azure [偵測資料集的資料漂移](https://docs.microsoft.com/zh-tw/azure/machine-learning/how-to-monitor-datasets?tabs=python#understand-data-drift-results)說明文件，說明可以在 Azure studio 的 \[資料集/監視器\] 頁面中更新設定，以及分析現有資料的特定時間週期，可深入解析到資料漂移的程度，以及要進一步調查的特徵與提示。
    -   ![](https://docs.microsoft.com/zh-tw/azure/machine-learning/media/how-to-monitor-datasets/drift-overview.png)
        -   圖片說明資料漂移的大小、資料集內最多漂移排序、閾值，資料來源:微軟 Azure [偵測資料集的資料漂移](https://docs.microsoft.com/zh-tw/azure/machine-learning/how-to-monitor-datasets?tabs=python#understand-data-drift-results)。

小結
--

-   定義資料的特徵 X 與標籤 y ，蒐集所需資料，建立達成的目標與基準警戒值，對後續監控與改進有幫助。
-   資料總是在變，世界也是不停往前走，可以監控資料飄移與偏斜，並關注 ML 系統的資安、隱私、公平的情形，改進特定/少數族群的服務體驗，也是 AI 落地必須要注意的。

參考
--

-   [Machine Learning Monitoring, Part 5: Why You Should Care About Data and Concept Drift  
    ](https://evidentlyai.com/blog/machine-learning-monitoring-data-and-concept-drift)
-   [Deeplearning.ai](https://read.deeplearning.ai/the-batch/issue-105/)
-   [What Is Data Drift?](https://streamsets.com/why-dataops/what-is-data-drift/)
-   [Introducing TensorFlow Data Validation: Data Understanding, Validation, and Monitoring At Scale](https://blog.tensorflow.org/2018/09/introducing-tensorflow-data-validation.html)
-   [Azure偵測資料集的資料漂移](https://docs.microsoft.com/zh-tw/azure/machine-learning/how-to-monitor-datasets?tabs=python#understand-data-drift-results)
