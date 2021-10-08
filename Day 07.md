# [Day 07 : MLOps 的挑戰與技術要求](https://ithelp.ithome.com.tw/articles/10260599)


###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10260599)


在 [Day 06](https://ithelp.ithome.com.tw/articles/10260304) 引用與介紹 3 個 MLOps 相關定義，如果 MLOps 是一種工程文化與實踐，旨在 ML 系統開發與 ML 系統操作，實際遇到的挑戰與技術要求 持續交付基金會 CDF 有不少深刻的整理，摘述如下:

MLOps 的挑戰
---------

-   [CDF MLOps SIG](https://cd.foundation/blog/2020/02/11/announcing-the-cd-foundation-mlops-sig/) 基於以下問題而關注 MLOps 發展:
    
    -   ML 使用持續交付 (CI/CD) 可能會提高部署速度，但是否會提高或確保品質？
    -   是否正在部署提供道德、公平和無偏見的預測模型？
    -   ML 執行的操作是否可驗證？
    -   能否嚴格維護和測試？
    -   是否有明確的 Metadata 收集、實驗跟蹤、版本控制、ETL 操作等指南？
-   [MLOps Roadmap 2021(草案)](https://github.com/cdfoundation/sig-mlops/blob/master/roadmap/2021/MLOpsRoadmap2021.md#what-is-mlops-not) 撰寫了各種 MLOps 挑戰與技術要求指引，摘述幾個特色說明:
    
    -   MLOps 實踐不限何種程式語言、平台、模組。
    -   在資料科學家可能不具有部署軟體服務的豐富經驗情況下，將訓練過的 ML 模型包裝為可部署的方法。
    -   將 MLOps 應用於 PB 級及更高級別的超大規模問題的方法。
    -   管理 MLOps 資產發布週期的治理流程，包括負責任的 AI 原則。
    -   模型的內在保護。
-   [MLOps Roadmap 2021(草案)](https://github.com/cdfoundation/sig-mlops/blob/master/roadmap/2021/MLOpsRoadmap2021.md#what-is-mlops-not) 的小叮嚀:
    
    -   必須認識到，雖然 Python 作為一種表達 ML 概念的語言很方便，但它是一種解釋性腳本語言，在生產環境中本質上是不安全的，因為任何可以注入 Python 環境的臨時 Python 源都可以不受約束地執行, 即使 shell 訪問被禁用。不應長期使用 Python 來構建任務關鍵型 ML 模型，而應採用更安全的設計選項。  
        ![/images/emoticon/emoticon82.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon82.gif)
-   在實踐 MLOps 過程中，以下四步驟可以供參考:
    
    1.  為了確保運行的實驗有信心建構出最佳模型，訓練資料、程式、模型需要追蹤程式、版本控制的一些技巧。
    2.  設定觸發器已重新運行訓練作業，通常是自動化的。
    3.  模型應該要經過嚴格的可控可逆的 CI/CD 流程進行適當的測試、評估和批准才能發布，此為資料科學家很大的缺口。
    4.  我們希望持續能理解模型的性能，這與確保品質與業務持續性很重要。
-   另外， Google Cloud 的 [Architecture for MLOps using TFX, Kubeflow Pipelines, and Cloud Build](https://cloud.google.com/architecture/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build) 指引，在意的是如何實現 MLOPs ，將 MLOps 說明為:
    
    -   如需在生產環境中結合機械學習系統，您需要安排 ML pipeline 中的步骤。此外，您需要將 pipeline 自動化，已持續訓練模型。如需實驗新想法與功能，您需要在 pipeline 採用 CI/CD 作法。

End to end 解決方案
---------------

MLOps 與 DevOps 不同之處，很大一部分在於紀錄的主體不同， MLOps 除了「持續整合」(CI)、「持續交付」(CD)，也需要實現「持續訓練」(CT)。  
![](https://i.imgur.com/LTotwUm.png)

-   上圖為 [Google](https://cloud.google.com/architecture/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build)展示的 CI/CD 流程與持續機械學習 CT 的關係:
    -   您的 Code 包含採用的"模型"，訓練後的"超參數"。
    -   您的資料是變動的資料。
    -   透過部署觸發持續訓練 CT 流程。
    -   最後部署在產品服務中。
-   基於用於生產的機械學習有龐大的自動化需求，簡要提出 Google 及 微軟的自動化解決架構如後。

### 端對端的 ML 流程 TensorFlow Extended (TFX)

-   Google 提出以 [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx?hl=zh-tw) 建構端對端的機械學習系統， TFX 主打用於生產情境的開源解決方案，從資料輸入、產生 Schema、資料驗證、前處理、調參、訓練到部署，系列文後續介紹。  
    ![](https://i.imgur.com/F6Ad1e0.png)

### Azure Machine Learning 的 Python 模型 MLOps

-   微軟 MLOps 系統解決方案， 在[使用 Azure Machine Learning 的 Python 模型 MLOps](https://docs.microsoft.com/zh-tw/azure/architecture/reference-architectures/ai/mlops-python)介紹也包含實作範例，相關資訊可參閱連結。  
    ![](https://i.imgur.com/xRrE2FN.png)

小結
--

-   MLOps 架構主要關係用於生產的機械學習系統，如何在市場中維持營運，並有系統的因應情境變化帶來的資料與模型調整作業，對自動化需求高。
-   既然 MLOps 是精神也是實踐，倡議時就不侷限在任何程式語言與平台。
-   [MLOps Roadmap 2021(草案)](https://github.com/cdfoundation/sig-mlops/blob/master/roadmap/2021/MLOpsRoadmap2021.md) 撰寫了各種 MLOps 挑戰與技術要求指引，雖然本系列文有摘述，礙於篇幅與知識量，仍建議有興趣者閱讀。

參考
--

-   [Announcing the CD Foundation MLOps SIG](https://cd.foundation/blog/2020/02/11/announcing-the-cd-foundation-mlops-sig/)
-   [MLOps Roadmap 2021(草案)](https://github.com/cdfoundation/sig-mlops/blob/master/roadmap/2021/MLOpsRoadmap2021.md)
-   [Architecture for MLOps using TFX, Kubeflow Pipelines, and Cloud Build](https://cloud.google.com/architecture/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build)
-   [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx?hl=zh-tw)
-   [使用 Azure Machine Learning 的 Python 模型 MLOps](https://docs.microsoft.com/zh-tw/azure/architecture/reference-architectures/ai/mlops-python)
