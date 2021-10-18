# [Day 23 : 模型分析 TensorFlow Model Analysis (TFMA)](https://ithelp.ithome.com.tw/articles/10269467)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10269467)


模型分析 TFMA 介紹
------------

-   過往我們關注模型的訓練結果，會追蹤該模型在每次 epochs 之後的 AUC 、 ACC、 loss 等指標變化，並且以視覺化繪圖方式呈現模型進展，此時擅長分析該模型狀況的 TensorBoard 就很好用。  
    ![](https://i.imgur.com/T2I6Ul0.png)
    
    -   圖片來源: [Azure 使用 TensorBoard 監視](https://docs.microsoft.com/zh-tw/visualstudio/ai/monitor-tensorboard?view=vs-2017)。
-   但是，如果要深入觀察模型「各版次」的狀況時，[TensorFlow Model Analysis (TFMA)](https://www.tensorflow.org/tfx/guide/tfma) 可以視覺化分析不同版次的模型狀況，讓您評估是否讓新模型更新上線，而不是把糟糕的模型替代原本的。
    
-   TFMA 在 TFX 自動化流程中實現的組件為 `ExampleValidator`，讓模型訓練完進行模型驗證，達到持續訓練的目的，續接完成自動部署模型。
    
-   更重要的是，模型驗證非常關心模型的「公平性」，善用模型分析工具能抓出模型弱點，進而回頭改進資料與模型。  
    ![](https://www.tensorflow.org/tfx/model_analysis/images/tfma-slicing-metrics-browser.gif)
    
-   TFMA 可以做到以下任務:
    
    -   根據整個訓練和保留數據集計算的指標，以次日的評估。
    -   隨時間跟蹤指標。
    -   用不同特徵切片分析模型性能。
    -   進行模型驗證。
-   TFMA 用來評估 TensorFlow 模型的程式庫，可搭配 TensorFlow 來建立 `EvalSavedModel` 做為分析的依據。使用者可透過這個程式庫，使用訓練程式中定義的相同指標，以分散的方式評估大量資料的模型。這些指標可根據不同的資料片段運算得出，並在 Jupyter 筆記本中以視覺化的方式呈現。
    

模型分析 TFMA 實作
------------

-   本篇將以官方範例示範，您可以跟著使用 [Colab 實作範例 ![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/tensorflow/tfx/blob/master/docs/tutorials/model_analysis/tfma_basic.ipynb#scrollTo=SA2E343NAMRF)，但因範例略顯臃腫，建議配合本文服用。

### 1\. 建立 TFMA 環境

-   安裝 `tensorflow-model-analysis`。
    
    -   模組須重啟 Colab 執行階段( Restart Runtime) 再執行後續操作。
        
        ```python
        # Upgrade pip to the latest, and install TFMA.
        !pip install -U pip
        !pip install tensorflow-model-analysis
        # Restart Runtime
        
        ```
        
-   下載資料集
    
    -   官方採用芝加哥市發布的[芝加哥計程車行程資料集](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)，資料集包含23個欄位，是個壓縮檔。
-   解析 Schema
    
    -   Schema 檔案源自 [TFDV](https://www.tensorflow.org/tfx/data_validation/)，如果沒有的話可以參見 [Day 14](https://ithelp.ithome.com.tw/articles/10263091) 介紹，將手上有的資料集透過 `tfdv.infer_schema` 產生及定義 Schema。
-   使用 Schema 建立 TFRecords 檔案
    
    -   配合 TFMA 輸入資料集格式 `*.tfRecords`，需撰寫程式將原始資料`data.csv` 轉換為符合 Schema 的資料型態( int、 float、str)、數據範圍檔案引入。
        -   範例中，新增 "big_tipper" 特徵以布林值表示小費是否大於票價20%。
    -   `*.csv` 轉為 `*.tfrecords` 的函數官方範例已寫好。

### 2\. 設置和執行 TFMA

-   TFMA 支持模型類型參見[格式介紹](https://www.tensorflow.org/tfx/model_analysis/get_started#model_types_supported)，包含 `TF keras`、基於 TF2 產生的 API 、 `tf.estimator` 、 `pd.DataFrame` 等類型。
-   此範例展示 `tf.keras` 模型、 `tf.estimator` 等 2 種做法供您參考，並分別保存為 `EvalSavedModel` 。
-   TFMA 支援在模型訓練期間的評估模型績效。
-   接著步驟為:
    -   設置 `tfma.EvalConfig`。
        -   在此設定哪個欄位是您的y標籤，評估機械時使用哪些指標，準備要產生哪些觀察切片。
    -   設置 `tfma.EvalSharedModel` 。
    -   使用 `tfma.run_model_analysis` 創建 `tfma.EvalResult` ，即可視覺化呈現模型績效。

### 3\. TFMA 視覺化模型績效

-   以 `tfma.view.render_slicing_metrics()` 視覺化呈現模型績效，您可以選擇想要觀察的切片、切換，此範例 `slicing_column='trip_start_hour'` 。  
    ![](https://i.imgur.com/5RneSfu.gif)
    
-   在圖表中:
    
    -   Visualization 可以切換2種樣貌，`Overview` 顯示每個切片，`Metrics Histogram` 是將結果分桶顯示。
    -   Examples (Weighted) Threshold 可以設定顯示的門檻值，超過門檻值才會顯示。
    -   Show 所呈現的觀察指標是您在 `tfma.EvalSharedModel` 時設置的，視需要可增減。
    -   示範中展示了某些時段 precision = 0， recall = 0 的狀況，透過 Sort 更清楚。
-   更多的嘗試
    
    -   例如替換切片欄位 'slicing\_column=trip\_start_day' 觀察。
        
    -   交叉組合觀察切片。
        
        ```python
        tfma.view.render_slicing_metrics(
            eval_result,
            slicing_spec=tfma.SlicingSpec(
                feature_keys=['trip_start_hour', 'trip_start_day']))
        
        ```
        
    -   設定`feature_values` 篩選特徵值。
        
        ```python
        tfma.view.render_slicing_metrics(
            eval_result,
            slicing_spec=tfma.SlicingSpec(
                feature_keys=['trip_start_day'], 
                feature_values={'trip_start_hour': '12'}))
        
        ```
        
        ![](https://i.imgur.com/iC6aipy.gif)
        
    -   另外也有 `tfma.view.render_plot` 顯示指定切片與觀察值，勾選 Show all plots 後，您可以看到非常豐富的視覺化圖表。  
        ![](https://i.imgur.com/2fyDzJZ.gif)
        

### 4\. 追蹤隨著時間推移的模型

-   在您訓練好模型，您會希望測試模型時使用生產情境產生的，畢竟那才是模型會遇到的真實反映。TFMA 可以幫助您持續監控與衡量模型性能。
-   先儲存儲存每個模型評估結果，範例展示了t0日到t2日的變化，在視覺化圖表中預設顯示AUC，您還可以新增比較圖。  
    ![](https://i.imgur.com/Jh6kghm.gif)

### 5\. 模型驗證

-   TFMA 可以同時評估多個模型，通常是比較基本模型與新模型之間的狀況，譬如可以鎖定新模型的 AUC 等績效要超過 Baseline，在設定好門檻值後，以 `tfma.ValidationResult` 查看驗證結果，如低於門檻值則驗證失敗。
    
    ```python
    validation_result = tfma.load_validation_result(validation_output_path)
    print(validation_result.validation_ok)
    # False
    
    ```
    

小結
--

-   本篇讓您對 TensorFlow Model Analysis (TFMA) 工具有更多的認識，驗證隨著時間推移的模型對於用於檢測已部署在生產情境的模型相當重要，TFMA 設計為可以直接產生驗證結果，也可以視覺化呈現。較不方便的是需要有 Schema 及設定 `tfma.EvalConfig` ，這也算是 TensorFlow 比較難以親近的風格吧。
-   希望能降低您使用 TFMA 工具的門檻，盡力採用 Gif 呈現，再不行要拍影片了...。  
    ![/images/emoticon/emoticon07.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon07.gif)

參考
--

-   [Azure 使用 TensorBoard 監視](https://docs.microsoft.com/zh-tw/visualstudio/ai/monitor-tensorboard?view=vs-2017)
-   [TensorFlow Model Analysis (TFMA)](https://www.tensorflow.org/tfx/guide/tfma)
