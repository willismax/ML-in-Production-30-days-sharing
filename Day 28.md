# [Day 28 : 用於生產的機械學習 TensorFlow Extended (TFX) 介紹](https://ithelp.ithome.com.tw/articles/10272958)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10272958)

什麼是 TensorFlow Extended (TFX)
-----------------------------

-   TensorFlow Extended (TFX) 是端對端平台，可部署於用於生產環境中的機器學習，建構及管理機械學習工作流程。依官方介紹，TFX提供下列三種主要功能:
    -   整合流程功能 - [TFX pipelines](https://www.tensorflow.org/tfx/guide?hl=zh-tw#understanding_tfx_pipelines) 可讓您自動化調度管理多個平台 (例如 Apache Airflow、Apache Beam 和 Kubeflow 管線) 上的機器學習工作流程，具有可攜性和互通性。
    -   積木功能 - [TFX 組件(components)](https://www.tensorflow.org/tfx/guide?hl=zh-tw#tfx_standard_components) 各有功能，可以組合流程使用。
    -   部分模組引入組件 - [TFX 部分模組 (libraries)](https://www.tensorflow.org/tfx/guide?hl=zh-tw#tfx_libraries) 可獨立做為模組或作為 TFX 組件使用。
    -   聽起來就像S.H.E.可以到處巡迴演唱、可以一起唱、單飛也很強大。

![](https://i.imgur.com/qjsjWcI.png)

TFX 的組成
-------

### 建立 TFX pipeline 的各元件 (component) 功能：

TFX 一系列元件，專門用於可擴充的高效能機器學習工作，包括建立模型、進行訓練、提供推論，以及管理線上、行動裝置 (TensorFlow Lite)和網頁應用服務 (TensorFlow JS) 的部署。

-   `ExampleGen`：
    
    -   流程管線的最初輸入元件，負責擷取輸入資料集，可以在此分割訓練、評估、驗證資料集。
    -   資料轉換為 `tf.Examples` 格式，提供訓練與評估之用。  
        ![](https://i.imgur.com/Banu4gk.png)
-   `StatisticsGen`：
    
    -   計算輸入的資料集的統計資料。
    -   類似`pandas.describe()`、`pandas.info()`的功能，也可以視覺化呈現。  
        ![](https://i.imgur.com/MwCPCaa.png)
-   `SchemaGen`：
    
    -   可檢驗統計資料並建立資料結構定義 Schema，像是資料型態、數值範圍、文字 Domain 等。
    -   Schema 也是驗證資料、特徵工程重要輸入之一。  
        ![](https://i.imgur.com/OZP43WM.png)
-   `ExampleValidator`：
    
    -   可查看資料集內是否有異常狀況和遺漏值，使用結構定義和統計資料，藉此查看資料中是否有異常情況、遺漏的值以及不正確的資料類型。
    -   上述4個元件屬於 TFDV 程式庫的功能，TFDV 會檢驗您的資料並推論資料類型、類別和範圍，然後自動協助您識別異常狀況和遺漏的值。  
        ![](https://i.imgur.com/axBpuFp.png)
-   `Transform`：
    
    -   這個元件會運用 [TensorFlow Transform (TFT)](https://www.tensorflow.org/tfx/guide/tft?hl=zh-tw) 程式庫的功能執行特徵工程，分散運算的底層採用 Apache Beam ，程式在部署情境會轉成`tf.Graph`，因為都是同一套程式碼，達到減少 `training-severing skew`。
        
    -   Transform 元件會產生 `SavedModel`，在 [Trainer](https://www.tensorflow.org/tfx/guide/trainer?hl=zh-tw) 元件執行期間匯入並用於 TensorFlow 中的建模程式碼。如果也產生了 `EvalSavedModel`，可進行後續模型分析，此時也須引入 TFMA 程式庫。
        
    -   參閱 [Day 15](https://ithelp.ithome.com.tw/articles/10263595)、[Day 16](https://ithelp.ithome.com.tw/articles/10264084) 說明功能。
        
        ![](https://i.imgur.com/4IRgyVm.png)
        
-   `Tuner`：
    
    -   可調整模型的超參數，可以在 Trainer 之前新增選用的 [Tuner](https://www.tensorflow.org/tfx/guide/tuner?hl=zh-tw) 元件，藉此調整模型的超參數 (例如層數)。有了指定的模型和超參數的搜尋空間，調整演算法會根據目標找出最佳超參數。
-   `Trainer`：
    
    -   訓練模型，就是 TensorFlow 的本體，`tensorflow`、`tensorflow.keras` 皆可。
-   `Evaluator`：
    
    -   可針對訓練結果執行深入分析，並協助您驗證匯出的模型，確保這些模型達到要求，可推送至生產環境。
    -   為了進行模型分析，需在有不停儲存各元件執行歷程的 [ML Metadata (MLMD)](https://www.tensorflow.org/tfx/guide/mlmd?hl=zh-tw) 找出這些元件的執行結果，透過 `tfma.load_eval_results`、`tfma.view.render_slicing_metrics` 可視覺化瞭解模型的特性，並視需要進行修改。
    -   對應獨立的模型分析 TFMA 模組，已在 \[Day 23\] 介紹。
-   `InfraValidator`：
    
    -   可檢查模型是否確實可從基礎架構提供，並避免推送未達到要求的模型。
    -   會啟動初期測試 TF Sever 模型伺服器以實際提供 SavedModel。如果通過驗證，
-   `Pusher`：
    
    -   可在提供服務的基礎架構上部署模型，元件最終會將 SavedModel 部署至 TFS 基礎架構，包括處理多個版本和模型更新。
    -   可用`Pusher`的功能分別對應部署在`TensorFlow Lite`、`TensorFlow JS` 、 `Tensorflow Serving` 情境。
-   `BulkInferrer`：
    
    -   可針對包含未標示推論要求的模型執行批次處理。
-   這些元件之間的資料流向如下圖所示：  
    ![](https://i.imgur.com/8RIemcS.png)
    
    -   資料來->`ExampleGen`接收 。
    -   `StatisticsGen` 接收 `ExampleGen` 數據產生統計情報，供 `SchemaGen` 產生 Schema、 `ExampleValidator` 驗證資料之用。
    -   `SchemaGen` 的資料定義也作為資料驗證`ExampleValidator` 、 `Transform` 特徵工程之用。
    -   `Transform` 特徵工程、`Tuner` 參數調整、`Trainer` 訓練模型為機械學習系統熟悉的建模核心。
    -   `Trainer` 訓練的模型交由 `Evaluator` 確保能否將模型投入生產、`InfraValidator` 內部驗證模型是否符合要求。
    -   經驗證後的模型，透過 `Pusher` 部署在網頁、手機終端設備、伺服器等情境。

### TFX 各程式庫 (libraries) 與元件的關係

-   TFX pipeline 各元件與 TFX 程式模組關係對應:  
    ![](https://i.imgur.com/7w9JQgB.png)
    
-   主要程式庫包含
    
    -   TensorFlow Data Validation (TFDV)。
        -   TFX 元件`ExampleGen`、`StatisticsGen`、`SchemaGen`、`ExampleValidator` 皆屬於 TFDV 程式庫的功能。
    -   TensorFlow Transform (TFT)。
    -   TensorFlow。
    -   KerasTuner。
    -   TensorFlow Model Analysis (TFMA)。
    -   TensorFlow Metadata (TFMD)：
        -   提供中繼資料的標準表示法，在使用 TensorFlow 訓練機器學習模型時相當實用。中繼資料可由手動產生，也可以在輸入資料分析期間自動產生，並可用於資料驗證、探索和轉換。中繼資料序列化格式包括：
    -   ML Metadata (MLMD)：
        -   用於記錄和擷取有關機器學習開發人員和數據資料學家工作流程的中繼資料。
        -   多數中繼資料都使用 TFMD 表示法，MLMD 使用 SQL-Lite、MySQL 以及其他類似的資料儲存庫來管理穩定性。

### 支援技術

-   [Apache Beam](https://www.tensorflow.org/tfx/guide/beam?hl=zh-tw) 是開放原始碼形式的整合式模型，用於定義批次和串流資料平行處理管線。TFX 使用 Apache Beam 來實作資料平行管線。然後，管線會由 Beam 支援的其中一個分散式處理後端執行，這些後端包括 Apache Flink、Apache Spark、[Google Cloud Dataflow](https://cloud.google.com/dataflow/?hl=zh-tw) 等等。
-   可選用Apache Airflow 和 Kubeflow 等自動化調度管理工具可讓您更輕鬆地設定、操作、監控及維護機器學習管線。
    -   [Apache Airflow](https://airflow.apache.org/)
        
        -   Apache Airflow 是一個平台，可透過程式輔助的方式編寫、排程及監控工作流程。TFX 使用 Airflow 將工作流程編寫為工作的有向非循環圖 (DAG)，而 Airflow 排程器會執行工作站陣列的工作，並且遵循指定的相依性。
            
        -   豐富的使用者介面方便您以視覺化的方式呈現在生產環境中執行的管線、監控進度，並視需要進行疑難排解。
            
        -   將工作流程定義為程式碼，即可更輕鬆地進行維護、建立版本、測試和協同合作。  
            ![](https://i.imgur.com/QlMxw8t.png)
            
            ![](https://airflow.apache.org/docs/apache-airflow/stable/_images/airflow.gif)
            
    -   [Kubeflow](https://www.kubeflow.org/)
        
        -   Kubeflow 旨在方便於 Kubernetes 中部署機器學習 (ML) 工作流程，並提高其可攜性與可擴充性。
        -   Kubeflow 的目標並非重新建立其他服務，而是讓您能以輕鬆直接的方式，將業界最佳的機器學習開放原始碼系統部署至不同基礎架構。
        -   [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/pipelines-overview/) 可讓您在 Kubeflow 上撰寫和執行可重現的工作流程，並整合實驗功能和筆記本式的體驗。
        -   Kubernetes 上的 Kubeflow Pipelines 服務包括託管中繼資料儲存庫、容器型自動化調度管理引擎、筆記本伺服器和使用者介面，可協助使用者大規模開發、執行及管理複雜的機器學習管線。Kubeflow Pipelines SDK 可讓您透過程式輔助的方式，來建立和共用管線的元件與組合。  
            ![](https://i.imgur.com/xVbVd1h.png)

小結
--

-   TFX 為了彈性可擴展的機械學習服務而生，切分多個模組串起機械學習流程，初步閱讀與理解較為複雜，希望整理後能讓您更容易理解。

參考
--

-   [TFX pipelines](https://www.tensorflow.org/tfx/guide?hl=zh-tw#understanding_tfx_pipelines)
-   [Apache Beam](https://blog.gcp.expert/apache-beam-dataflow/)
-   [TensorFlow Extended (TFX) 概述和預訓練工作流程（TF Dev Summit '19） 上](https://www.youtube.com/watch?v=A5wiwT1qFjc)、[下](https://www.youtube.com/watch?v=0O201IQlkxc)
-   [TFX Airflow Tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/airflow_workshop/?hl=zh-tw)
