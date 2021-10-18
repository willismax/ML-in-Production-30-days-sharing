# [Day 30 : 綜合整理 MLOps Level 0 ~ 2](https://ithelp.ithome.com.tw/articles/10274317)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10273652)



-   MLOps 是值得持續投入的新興學門，如同 Day 01 談到的此系列目的，談如何從佈署機械學習至商業情境(ML in Production)，並關注佈署之後所需注意的資料品質、模型版本控制與剪枝、AI 可解釋力、錯誤分析、自動化 ML 到持續佈署，期待用 ML 專案生命週期的角度執行 MLOps 需要的。目標很宏大，篇幅與表達能力有限，筆者很享受這趟整理的路程。
    
-   30 天過去了，梳理我們在AI落地談 MLOps 中有那些進展。
    

### 談觀念:

-   此階段闡述 MLOps 精神與旨趣，呼應「以資料為中心的人工智慧」的想法，將用於生產的機械學習應有做為，以機械學習系統的生命週期解構從範疇、資料、建模到佈署應注意的人與事。
    -   [Day 01 : 這系列文在做什麼-緣起](https://ithelp.ithome.com.tw/articles/10258837)
    -   [Day 02 : 用於生產的機械學習 ML in Production](https://ithelp.ithome.com.tw/articles/10258861)
    -   [Day 03 : ML in Production 的挑戰](https://ithelp.ithome.com.tw/articles/10259314)
    -   [Day 04 : 以資料為中心的人工智慧 Data - Centric AI](https://ithelp.ithome.com.tw/articles/10259708)
    -   [Day 05 : ML 專案生命週期](https://ithelp.ithome.com.tw/articles/10259989)
    -   [Day 06 : 什麼是 MLOps](https://ithelp.ithome.com.tw/articles/10260304)
    -   [Day 07 : MLOps 的挑戰與技術要求](https://ithelp.ithome.com.tw/articles/10260599)
    -   [Day 08 : ML 工程師職責與分工](https://ithelp.ithome.com.tw/articles/10260962)
    -   [Day 30 : 綜合整理 MLOps Level 0 ~ 2](https://ithelp.ithome.com.tw/articles/10274317)

### 談實踐

技術介紹隨著機械學習生命週期推進，系列主軸希望介紹由 Google 開源的 TensorFlow Extended (TFX) 做為用於生產的機械學習框架，也以ML系統生命週期介紹實踐與模型優化方法。

-   範疇領域 Scope
    -   [Day 09 : 用於生產的機械學習 - 定義範疇 Scope](https://ithelp.ithome.com.tw/articles/10261352)
-   資料領域 Data
    -   [Day 10 : 用於生產的機械學習 - Data Define 與建立基準](https://ithelp.ithome.com.tw/articles/10261664)
    -   [Day 11 : 用於生產的機械學習 - Data Labeling 資料標註](https://ithelp.ithome.com.tw/articles/10262021)
    -   [Day 12 : 弱監督式標註資料 Snorkel (spam 入門篇)](https://ithelp.ithome.com.tw/articles/10262325) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1mgroggxRG_yuLw2OaBD8OBzDuBQ57ZTo?usp=drive_fs)
    -   [Day 13 : 弱監督式標註資料 Snorkel (視覺關係偵測篇)](https://ithelp.ithome.com.tw/articles/10262699) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1WsbfDk9r_g9_75CuQNmkvn3GzDhExmjc)
    -   [Day 14 : 資料驗證 TensorFlow Data Validation (TFDV)](https://ithelp.ithome.com.tw/articles/10263091) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/data_validation/tfdv_basic.ipynb)
    -   [Day 15 : 特徵工程 tf.Tramsform 介紹](https://ithelp.ithome.com.tw/articles/10263595)
    -   [Day 16 : 特徵工程 tf.Tramsform 實作](https://ithelp.ithome.com.tw/articles/10264084) [官方 ![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/tensorflow/tfx/blob/master/docs/tutorials/transform/simple.ipynb)
    -   [Day 17 : 用於生產的機械學習 - 特徵選擇 Feature Selection](https://ithelp.ithome.com.tw/articles/10264846) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1Y37iCwCCaaSg8U-mHrETtqWf3IPn2b9V?usp=drive_fs)
-   建模領域 Modeling
    -   [Day 18 : 深度學習(神經網絡)自動調參術 - KerasTuner](https://ithelp.ithome.com.tw/articles/10265801) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1qkot7-OLWTf9F5SyaS0_6doBKCy33qM5?usp=drive_fs)
    -   [Day 19 : 深度學習(神經網絡)自動建模術 - AutoMLs](https://ithelp.ithome.com.tw/articles/10266499) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/keras-team/autokeras/blob/master/docs/ipynb/image_classification.ipynb)
    -   [Day 20 : 模型優化 - 訓練後量化 Post Training Quantization](https://ithelp.ithome.com.tw/articles/10267328) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1ukgVrMdtWjpReIygWHJ7-Lcw61Lv5kAO)
    -   [Day 21 : 模型優化 - 剪枝 Pruning](https://ithelp.ithome.com.tw/articles/10268124) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1QQ0rZ9f18APlBy23M3-hfTMb4a5LHFtw)
    -   [Day 22 : 模型優化 - 知識蒸餾 Knowledge Distillation](https://ithelp.ithome.com.tw/articles/10268783) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1R1EQrUEP2Sb5gq-dIf_wbyA5KOhtRBWv)
    -   [Day 23 : 模型分析 TensorFlow Model Analysis (TFMA)](https://ithelp.ithome.com.tw/articles/10269467)[![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/tensorflow/tfx/blob/master/docs/tutorials/model_analysis/tfma_basic.ipynb#scrollTo=SA2E343NAMRF)
    -   [Day 24 : 負責任的 AI - Responsible AI (RAI)](https://ithelp.ithome.com.tw/articles/10270241)
    -   [Day 25 : 可解釋的 AI - Explain AI (XAI)](https://ithelp.ithome.com.tw/articles/10270902)
    -   [Day 26 : 公平指標與實作 Fairness Indicators](https://ithelp.ithome.com.tw/articles/10271626) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_TFCO_CelebA_Case_Study.ipynb#scrollTo=GRIjYftvuc7b)
-   部署領域 Deploying
    -   [Day 27 : 使用 TensorFlow Serving 部署 REST API](https://ithelp.ithome.com.tw/articles/10272257) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1-9cCg9xhWQb7itAcLhko8UqpNOFnu-nx)
    -   [Day 28 : 用於生產的機械學習 TensorFlow Extended (TFX) 介紹](https://ithelp.ithome.com.tw/articles/10272958)
    -   [Day 29 : 用於生產的 TensorFlow Extended (TFX) 實作](https://ithelp.ithome.com.tw/articles/10273652) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1o4lRoAdpPkfCL6WV3X6JwXK5C27itbI6?usp=drive_fs)

MLOps Level 0 ~ 2
-----------------

-   系列文的最後，容許筆者將 MLOps 等級劃分作為投入用於生產的機械學習的實踐路徑。以下的等級劃分採用 Google 的定義，採等級 0 至 2 ，微軟也有[機器學習作業成熟度模型](https://docs.microsoft.com/zh-tw/azure/architecture/example-scenario/mlops/mlops-maturity-model)的定義分為等級 0 至 5 ，您可以相互對照參考。

### MLOps Lebel 0 : 手動的過程

![](https://cloud.google.com/architecture/images/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-2-manual-ml.svg)

-   一個全手動，沒有 CI/CD 的過程，訓練與佈署之間可能有時間延遲的偏態 Skew 產生，非自動化持續訓練，版本追溯有難度，也缺乏主動監控機制，在最初投入產生可行方案時的過渡時期，但較難因應服務崩潰與中斷的狀況。

### MLOps Lebel 1 : 自動化流程

![](https://cloud.google.com/architecture/images/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-3-ml-automation-ct.svg)

-   MLOps 等級 1 導入自動化訓練流程，可以自動持續交付佈署模型，另外也有手動訓練模型的機制，訓練好的模型交付 IT 或維運團隊佈署上線。
-   等級 1 已經足夠挑戰用於生產的情境，模型需手動測試管道與組件，在不頻繁的佈署、尚只有少數機械學習服務投入生產情境時還可以因應管理，一但您在生產中管理許多 ML 服務，則需要一個 CI/CD 設置來自動構建、測試和部署 ML 的流程。

### MLOps 級別 2：CI / CD 管道自動化

![](https://cloud.google.com/architecture/images/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-4-ml-automation-ci-cd.svg)

-   為了快速可靠地更新生產中的管道，您需要一個強大的自動化 CI/CD 系統。這個自動化的 CI/CD 系統讓資料科學團隊快速探索特徵工程、模型架構和超參數的新想法。並且可以實現自動構建、測試並將新的管道組件部署到目標環境。
-   下圖為具有自動觸發、 CI/CD/CT 與監控的自動化機械學習工作流程。  
    ![](https://cloud.google.com/architecture/images/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-5-stages.svg)

### 致謝

-   今年為了 MLOps 做了許多功課，很大一部分要感謝 GCP 的活動 [Google Cloud 開發者技術培訓計劃 / Google Cloud Study Jam (Taiwan and Hong Kong)](https://www.facebook.com/groups/googlecloudstudyjamtwhk)，解完12組題組(其實早就超過)拿到衣服、背包，又能實際摸索 GCP 各功能，相當優質。
-   更要感謝 Coursera 開設的 [Machine Learning Engineering for Production (MLOps) ](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)課程，當得知吳恩達新課剛好是期待的 MLOps 主題，立馬申請助學金完成這4堂優質技術內容，每門課程約需進行4至5週，總是迫不及待想整理與分享，也因此能有更多有趣實用的主題能整理至鐵人賽。
-   本系列範例多以官網或官網修改而成，原因為對照官方文件學習資源較為充足，即便如此官方範例實在是略顯繁雜，特別是以 TFX 介紹為主軸不是件容易的事。適逢 2021年5月19 TFX 正式邁入 1.0 穩定版，深感 TFX 中文資源缺乏，於是有了在 iThome 鐵人賽貢獻與整理的念頭。
-   最後就是感謝能陪我一起煎熬的走完 30 日內容的您，有限篇幅內講述龐大知識體系著實不易，興許會隨著 MLOps 主題延伸介紹更新，如果對您有幫助就值得了，由衷感謝。

![/images/emoticon/emoticon41.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon41.gif)![/images/emoticon/emoticon41.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon41.gif) ![/images/emoticon/emoticon41.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon41.gif) ![/images/emoticon/emoticon41.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon41.gif) ![/images/emoticon/emoticon41.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon41.gif)

參考
--

-   [MLOps：機器學習中的持續交付和自動化流水線](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
-   [機器學習作業成熟度模型](https://docs.microsoft.com/zh-tw/azure/architecture/example-scenario/mlops/mlops-maturity-model)
-   [Google Cloud 開發者技術培訓計劃 / Google Cloud Study Jam (Taiwan and Hong Kong)](https://www.facebook.com/groups/googlecloudstudyjamtwhk)
-   [Machine Learning Engineering for Production (MLOps)](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)
