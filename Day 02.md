# [Day 02 : 用於生產的機械學習 ML in Production](https://ithelp.ithome.com.tw/articles/10258861)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10258861)



ML 就像孩子一樣，孩提時百般呵護，長大時不得不面對外界的殘酷。佈署到商務情境的 ML 模型，某方面像是放飛的孩子，既期待又怕受傷害，如果沒有持續的引導、學習、檢核、導正他，不慎走歪路時就會是父母及社會的痛，付出更大的成本與代價。  
![/images/emoticon/emoticon10.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon10.gif)

為什麼我們要在意「用於生產的機械學習」
-------------------

-   自從 AlphaGo 在 2016 年大戰圍棋冠軍事件橫空出世， AI 人工智慧領域的 ML 機械學習、 DL 深度學習蓬勃發展，至今走進實際應用階段，演算法創新與突破帶來「產業人工智慧化」、「人工智慧產業化」的契機，也見到許多 AI 新創巨星殞落。 [deeplearming.ai](https://info.deeplearning.ai/the-batch-companies-slipping-on-ai-goals-self-training-for-better-vision-muppets-and-models-china-vs-us-only-the-best-examples-proliferating-patents) 引述資料指出，研究發現，儘管人工智能預算在增加，但只有 22% 的使用機器學習的公司成功部署了模型。換言之，八成以上的企業無法成功部署機械學習模型，問題出在哪裡?
-   許多企業意識到，公開研究成果的 ML/DL 模型足夠優異，開源可以直接取用，但以自身數據訓練模型時，效果往往比 COCO 、 ImageNet 等公開資料集下降許多，而且燒錢不符成本。歸諸問題點，也許是模型算法的問題、調整超參數的問題、或是資料集的問題、人才的問題...，也難為資料科學團隊， AI 落地真的不是件容易的事情。
-   如果您開始覺得這是個問題，應該能接受要用更宏觀的角度思考 ML 生命週期，以及關注訓練模型以外的那些事。

用於生產中的機械學習 ML in Production
---------------------------

-   當您將AI佈署到商務情境之中，後續的營運、監控有不可忽視的重要性。在機械學習、深度學習有幾位活躍的指標人物， Google Brain 創辦人吳恩達絕對是其中之一，吳恩達於 2017 年成立 [Landing.ai](https://landing.ai/) 公司提供企業縮減 AI 商業化跟概念驗證等服務。同年亦成立 [Deeplearning.ai](https://www.deeplearning.ai/) 投入人工智慧領域專業技術教學，於Coursera開設諸多乾貨滿滿的課程(也是共同創辦人)，終於在今年6月推出 [Machine Learning Engineering for Production (MLOps) 系列課程](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops) ，為「用於生產中的機械學習」提出架構性見解，期待將拚種研究與發明 AI 模型 "Modele-centric AI" 的焦點風氣，轉向同樣重要的 "Data-centric AI" ，以資料中心的方法，更系統化的回推模型範疇、資料、建模訓練與佈署該注意的事。
-   站在巨人的肩膀上可以看得更遠，本系列文將以吳恩達提出的 ML 專案生命週期為架構，摘述並補充可行的設計模式與實務見解。

ML Engineering for Production (MLOps)
-------------------------------------

-   MLOps 是實踐 AI 落地的較高效的方法，畢竟數百萬人級的服務沒有自動化處理是不切實際的作法，故在討論 ML in Production 時，也會觸及到 MLOps ，如果能有建構自動化 ML Pipeline ，從資料蒐集、清洗、建模、驗證到佈署一系列都搞定更好。在 [ML Ops: Machine Learning as an Engineering Discipline](https://towardsdatascience.com/ml-ops-machine-learning-as-an-engineering-discipline-b86ca4874a3f) 一文也闡述了認為 MLOps 可做為一學門潛力。
-   既然 ML　Pipeline 是複雜的，要兼顧彈性可拆、可控且自動化，系列文會帶出 MLFLow 、 AutoML ，會介紹　[TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx?hl=zh-tw) ，其中 TFX 為了兼顧上述需求，形成了相當龐大的系列家族，將各個細目的拆成組件，且可獨立運用，今年終於發布為 1.0 版，並且持續發展中。
    
    -   ![](https://i.imgur.com/AsxctPO.png)
    
    > [TFX Pipeline](https://www.tensorflow.org/tfx?hl=zh-tw) 是實作機器學習管線的一系列元件，專門用於可擴充的高效能機器學習工作，元件是使用 TFX 程式庫所建構，而這些程式庫也可以分開使用。
    
-   微軟 Azure 亦相當重視[機器學習作業 (MLOps)](https://azure.microsoft.com/zh-tw/services/machine-learning/mlops/#features)，文章也會適時說明 Azure 的方案。
    
    -   ![](https://github.com/microsoft/MLOps/raw/master/media/ml-lifecycle.png)
    
    > [MLOps on Azure](https://github.com/microsoft/MLOps) 可讓資料科學和 IT 小組透過監視、驗證及治理機器學習模型，來共同作業並增加模型開發和部署的步調。
    
-   最後，佈署機械學習 ML 的工程實務逐漸歸納並越受重視，也多虧軟體工程 DevOps 精神興起與 CI/CD 實務越漸普及，本系列期待透過鐵人賽將系列知識做個梳理，拋磚引玉，協助銜接 ML 商務落地的一哩路，以及對 AI 有志趣者，補充除了 AI 建模以外需要關注的面向。

小結
--

-   本日想說明的是，企業將 AI 模型推進至業務情境並持續維運，是相當不容易的，在此也向企業及以此為職志的資料/維運團隊致上崇高敬意。越早具有成熟經驗的企業將取得先機，無法踏入的企業拉鋸逐漸擴大，因為改進模型成效的動能來自數據，變動的數據透過異常分析可以改善品質。
-   MLOps 關注模型成效至部署/維運階段的任務，是工作職務也是合作精神，結合版本控制、 DevOps 等軟體工程可以較有效的完成 ML 部署/維運任務。

參考
--

-   [deeplearming.ai電子報 - 2019/12/18](https://info.deeplearning.ai/the-batch-companies-slipping-on-ai-goals-self-training-for-better-vision-muppets-and-models-china-vs-us-only-the-best-examples-proliferating-patents)
-   [Landing.ai](https://landing.ai/)
-   [Deeplearning.ai](https://www.deeplearning.ai/)
-   [Machine Learning Engineering for Production (MLOps) 系列課程](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)
-   [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx?hl=zh-tw)
-   [機器學習作業 (MLOps)](https://azure.microsoft.com/zh-tw/services/machine-learning/mlops/#features)
