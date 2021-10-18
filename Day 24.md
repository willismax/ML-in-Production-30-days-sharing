### [Day 24 : 負責任的 AI - Responsible AI (RAI)](https://ithelp.ithome.com.tw/articles/10270241)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10270241)


-   當您用心呵護的機械學習終於實現，期待能滿足與提升使用者福祉，您應該有足夠的信心與能力對產品負責， AI 產品亦然。
-   延續系列文對您的機械學習產品生命週期的思考，在資料驗證的任務可用 TFDV 視覺化、自動化檢視資料的狀況 ([Day 14](https://ithelp.ithome.com.tw/articles/10263091)) ，在模型訓練完成可透過模型分析 TFMA 剖析模型在不同主題切片之下的預測成果 ( [Day 23](https://ithelp.ithome.com.tw/articles/10269467))，現在與您分享驅動資料驗證與模型分析的核心思考 \- 負責任的 AI (Responsible AI)。

什麼是 Responsible AI？
-------------------

-   人工智慧與您的機械學習系統是用以解決現實生活問題的，談論更有效的建構可以造福所有人的 AI 系統，即是 Responsible AI 的精神。
    
-   Responsible AI (RAI) 是現今對機械學習服務的深刻思考，認為各種組織在發展機械學習服務時，應重視服務對象的個體性與特殊性。鑒於資料集的偏斜可能導致預測結果偏頗，諸如升遷預測系統的資料集側重單一性別以致升遷選拔帶有性別定見，人臉生成系統未考慮種族多樣性以致偏向當一種族，您原本視為依據數據理所當然的產出也應考慮以人為本的公平性。再者，隨著隱私權的議題越見重視，[歐盟一般資料保護法規 General Data Protection Regulation (GDPR)](https://zh.wikipedia.org/wiki/%E6%AD%90%E7%9B%9F%E4%B8%80%E8%88%AC%E8%B3%87%E6%96%99%E4%BF%9D%E8%AD%B7%E8%A6%8F%E7%AF%84) 揭示用戶有權請求被收集的資料，且有權在特定情況下刪除其資料。身為 AI 服務提供者，您如何在訓練情境中抽離個資議題去識別化，以及能在各個版本控制中刪除特定使用者的數據。面對 AI 服務攻擊的惡意，監測與回溯版本控制，解釋與理解模型現況，諸多議題匯聚為 Responsible AI 討論範疇，也成為實踐 AI 服務的顯學與課題，比起實踐的技術，衷於所有人福祉的服務更加重要。
    
-   負責任的 AI 關注焦點引述 [Google Responsible AI](https://www.tensorflow.org/responsible_ai) 說明，包含：
    
    -   **AI 最佳實踐 Recommended best practices for AI**
        -   設計 AI 系統時應遵循軟體開發最佳做法，並且採用「以人為本」的方法運用機器學習技術。
    -   **公平性 Fairness**
        -   隨著 AI 對於各領域和社會的影響逐漸增加，建立「公平」且「可包容所有人」的系統至關重要。
    -   **可解釋性 Interpretability**
        -   我們必須「瞭解及信任」 AI 系統，才能確保系統能夠如預期般運作。
    -   **隱私權 Privacy**
        -   利用機密資料訓練模型時，需要「妥善保護隱私權」。
    -   **安全性 Security**
        -   「辨識潛在威脅」有助於維持 AI 系統安全。
-   上述的「以人為本」、「公平」、「包容」、「瞭解及信任」、「妥善保護隱私權」、「辨識潛在威脅」串起您的機械學習系統實踐負責任的 AI 方針。
    

御三家的 Responsible AI 發展
----------------------

-   Google
    
    -   Google 在 2018 年推出了 [AI 原則](https://www.blog.google/technology/ai/ai-principles/)，之後，TensorFlow 等團隊依該原則發展產品，也推出 Responsible AI (RAI) 做法的工具和技巧，成為相當龐大的知識體系。
-   Azure
    
    -   Azure 提出 [Responsible ML](https://docs.microsoft.com/zh-tw/azure/machine-learning/concept-responsible-ml)，並且在文章中介紹諸多工具。
        -   在解讀和說明模型行為提出 [InterpretML](https://github.com/interpretml/interpret) ，是由 Microsoft 所建置的開放原始碼解釋工具。
        -   減少不公平的工具為 [FairLearn](https://github.com/fairlearn/fairlearn) ，也是開放原始碼套件。
        -   實作差異隱私系統十分困難，開放原始碼專案 [SmartNoise](https://github.com/opendifferentialprivacy/smartnoise-core) 可以協助完成任務。  
            ![](https://i.imgur.com/qgkwLI3.png)
-   AWS
    
    -   在 AWS Marketplace 有諸多第三方 Responsible AI 服務，諸如[Dataiku](https://www.dataiku.com/) 有相關功能，[Amazon SageMaker Clarify](https://aws.amazon.com/tw/sagemaker/clarify/)也包含相對應功能，幫助您更能掌握模型。

### 機器學習工作流程中的 Responsible AI

-   負責任的 AI 以機械學習服務生命週期各階段任務須注意事項，本篇延續[Google 說明](https://www.tensorflow.org/responsible_ai):
-   **1\. 界定問題 Define problem**
    -   我的機器學習系統是為誰而設計？
        
        > 使用者實際體驗系統的方式，對於評估機器學習系統預測、建議及決策的真實作用十分重要。請務必在開發過程中，儘早取得各類型使用者提出的意見。
        
    -   相關工具
        -   [People + AI (PAIR) Guidebook](https://pair.withgoogle.com/guidebook/) 是相當有參考價值的提供服務指引。
        -   [PAIR Explorables](https://pair.withgoogle.com/explorables/) 可以直接探索 PAIR 主題。
-   **2\. 建購及準備資料 Construct & prepare data**
    -   我使用的資料集是否具有充分代表性？
        
        > 你的資料取樣方式是否可代表使用者 (例如：將用於所有年齡層，但你只有銀髮族的訓練資料) ，以及是否符合現實環境情況 (例如：將使用一整年，但你只有夏季的訓練資料)？
        
    -   我的資料中存在現實環境偏誤/人類認知偏誤嗎？
        
        > 資料中的潛在偏誤可能會形成複雜的回饋循環， 加深既有的刻板印象。
        
    -   相關工具
        -   [Know Your Data](https://knowyourdata.withgoogle.com/)
        -   [TensorFlow Data Validation (TFDV)](https://www.tensorflow.org/tfx/guide/tfdv)
        -   [Data Card](https://research.google/static/documents/datasets/crowdsourced-high-quality-colombian-spanish-es-co-multi-speaker-speech-dataset.pdf) 可以將資料集以 HTML 圖卡呈現。
-   **3\. 建購及訓練資料 Build & train model**
    -   我應該使用什麼方法訓練模型？
        
        > 使用可在模型中建構公平性、可解釋性、隱私和 安全性的訓練方法。
        
    -   相關工具
        -   [Model Remediation](https://www.tensorflow.org/responsible_ai/model_remediation)
        -   [TensorFlow Privacy](https://www.tensorflow.org/responsible_ai/privacy/guide)
        -   [TensorFlow Federated：分散式資料的機器學習](https://www.tensorflow.org/federated)
        -   [TensorFlow Constrained Optimization (TFCO)](https://github.com/google-research/tensorflow_constrained_optimization/blob/master/README.md)
        -   [TensorFlow Lattice (TFL)](https://www.tensorflow.org/lattice/overview)
-   **4\. 評估模型 Evaluate model**
    -   我的模型成效如何？
        
        > 針對現實環境裡的各類 使用者、用途和使用情境，評估使用者體驗。
        
    -   相關工具
        -   [Fairness Indicators](https://www.tensorflow.org/responsible_ai/fairness_indicators/guide) 公平指引將獨立一篇介紹。
        -   [TensorFlow Model Analysis (TFMA)](https://www.tensorflow.org/tfx/model_analysis/install)
        -   [What-If Tool](https://pair-code.github.io/what-if-tool/) 以假設性的問題視覺化探索主題。
        -   [Language Interpretability Tool](https://www.tensorflow.org/responsible_ai)
        -   [Explainable AI](https://cloud.google.com/explainable-ai)
        -   [TensorFlow Privacy](https://blog.tensorflow.org/2020/06/introducing-new-privacy-testing-library.html)
        -   [TensorBoard](https://www.tensorflow.org/tensorboard/get_started)
-   **5\. 部署及監控 Deploy & monitor**
    -   是否存在複雜的回饋循環？
        
        > 即使整體系統設計經過悉心規劃， 以機器學習為基礎的模型在套用到 真實的動態資料時，很少能夠完美運作。當實際使用的產品發生問題時，請思考 該問題是否與任何弱勢族群議題相呼應，並考量 短期和長期解決方案對於該問題的影響。
        
    -   相關工具
        -   [Model Card 工具包](https://www.tensorflow.org/responsible_ai/model_card_toolkit/guide)
        -   [ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd)
        -   [Model Cards](https://modelcards.withgoogle.com/about)

PAIR : People + AI Research
---------------------------

![](https://i.imgur.com/1fHNW1l.png)

-   People + AI (PAIR) 是一組使用 AI 進行設計的方法、最佳實踐和示例，建議閱讀，裡面包含您的模型服務如何設計的指引。譬如資料的呈現、按鈕的文案，增加使用者信任感的設計、如何有禮貌地拒絕與導引等，可以讓您的商業服務更順遂。
-   譬如[對解釋性-信任](https://pair.withgoogle.com/chapter/explainability-trust/)的解釋，提出 AI 服務取得用戶信任的主要因素:
    -   **能力** 是產品完成工作的能力，力求提供易於識別的有意義的價值的產品。
    -   **可靠** 表明您的產品如何始終如一地發揮其能力。根據用戶設定的期望，它是否符合品品質標準？僅當您能夠滿足您已設置並向用戶透明地描述的欄時才啟動。
    -   **仁慈** 是相信受信任方希望為用戶做好事的信念。用戶從他們與你的產品的關係中得到了什麼，你又從中得到了什麼？對此要誠實和坦率。  
        ![](https://i.imgur.com/FRzg1eY.png)
    -   上圖為 PAIR 實際舉例，可以改進您 AI 服務體驗，也有不少可以上您 AI 服務到位的建言。

小結
--

-   一整理下來就頭昏了，您可以感受到近期隨著對 ML/DL 模型解釋力、公平性的需求，各家平台以及開源工具著墨甚多，也是近期人工智慧領域發展的核心，用系統性的從定義、資料、模型到佈署的負起建構 AI 服務應有的責任與使命，讓所有人都能有所裨益。
-   其中不乏酷酷的東西，後續將會介紹 AI 解釋力與演示公平性的調整實作。

![/images/emoticon/emoticon41.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon41.gif)

參考
--

-   [瞭解如何使用 TensorFlow，將 Responsible AI 的做法整合至機器學習工作流程](https://www.tensorflow.org/responsible_ai)
-   [歐盟一般資料保護法規 General Data Protection Regulation (GDPR)](https://zh.wikipedia.org/wiki/%E6%AD%90%E7%9B%9F%E4%B8%80%E8%88%AC%E8%B3%87%E6%96%99%E4%BF%9D%E8%AD%B7%E8%A6%8F%E7%AF%84)
-   [Responsible ML](https://docs.microsoft.com/zh-tw/azure/machine-learning/concept-responsible-ml)
-   [PAIR](https://pair.withgoogle.com/)
-   [隨機響應如何幫助負責任地收集敏感信息](https://pair.withgoogle.com/explorables/anonymization/)
