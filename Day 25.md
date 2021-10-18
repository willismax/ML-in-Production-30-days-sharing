# [Day 25 : 可解釋的 AI - Explain AI (XAI)](https://ithelp.ithome.com.tw/articles/10270902)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10270241)


-   AI 黑箱作業已經被詬病許久，因為 AI 類神經網絡的複雜性不似機械學習的樹狀結構、線性結構容易理解中間判斷過程，但隨著可解釋 AI 技術的出現，理解模型可以協助用於生產的機械學習系統有更佳的解釋能力。
    
-   在 2016 年有研究以 LIME 技術得知，訓練出的狼與哈士奇分類器，其實只是判別背景為雪地與否([簡報](https://filene.org/assets/images-layout/Panel_Singh.pdf)、[論文](https://arxiv.org/pdf/1602.04938.pdf))，李弘毅老師也做了數碼寶貝、寶可夢分類研究，經解釋原來是`*.png` 對透明背景處理為黑色，與`*.jpg`圖片的白色背景的差異被 AI 作為判斷依據([簡報](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/xai_v4.pdf))。
    
-   在用於生產的機械學習情境，可解釋的 AI 用來做為更深刻的討論用途，XAI 主要希望是
    
    -   確保演算法公平。
    -   鑑別出訓練資料潛在偏見與問題。
    -   確保演算法與模型符合預期。
-   解釋黑箱來鑑別與修正模型的問題才是 XAI 旨趣，特別是公平與偏見問題，存在性別、種族等偏見問題的 AI 是不允許上線的，已上線的 AI 服務在接收新資料持續訓練的過程被帶壞了，甚至面對針對 AI 服務的惡意攻擊，多用點方法解釋黑箱就有必要。
    

SHAP
----

-   [SHAP（SHApley Additive exPlanations）  
    ](https://github.com/slundberg/shap)可以解釋模型特徵之間影響力的模型，主要結合博弈理論與局部解釋力，也可以視覺化呈現解釋成果。
    
-   可以解釋的工具包含`TreeExplainer`、`DeepExplainer`、`GradientExplainer`、`KernelExplainer`，對應不同模型的解釋器。
    
-   以下圖片紅色表示正相關的力道、藍色表示負相關的力道，各特徵互有拉鋸影響力，可以看出何為關鍵特徵及其影響程度。  
    ![](https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_header.svg)  
    ![](https://i.imgur.com/RfWJtC3.png)  
    ![](https://i.imgur.com/7DeFhTi.png)
    
-   影像的分類重要性也可以視覺化得知AI主要判斷依據，進而解釋模型。  
    ![](https://i.imgur.com/2UuPqDF.png)  
    \- 圖片來源: [SHAP](https://colab.research.google.com/github/slundberg/shap/blob/master/notebooks/image_examples/image_classification/Multi-class%20ResNet50%20on%20ImageNet%20(TensorFlow).ipynb#scrollTo=VIRIhELocwYL)
    
-   以下為使用 SHAP 將[Fashion MNIST](https://keras.io/api/datasets/fashion_mnist/)資料集，以 CNN 訓練完的各特徵解釋對照結果。
    
    -   ![](https://i.imgur.com/bRzZrex.png)
        
    -   該模型對這 10 個類別分類，對角線紅色居多，表示能使其正確預測的主要因素，您可以仔細關注某些分類，譬如第 5 個分類 Coat 與第 3 分類 Pullover、 第 7 分類Shirt 有些形似，紅色、藍色的點都有出現，似乎也有理由讓模型判別為 Pullover 或 Shirt 。
        

小結
--

-   可解釋的 AI 技術本篇引用 SHAP 做說明，僅是諸多解釋工具之一， [LIME](https://github.com/marcotcr/lime) 、 IG 等各有適用情境
-   賦予提供「負責任的 AI 」解釋能力，該用途最終回歸到提供可靠、可信任、公平的機械學習服務。  
    ![/images/emoticon/emoticon42.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon42.gif)

參考
--

-   [Why Should I Trust You? Explaining the Predictions of Any Classifier 簡報](https://filene.org/assets/images-layout/Panel_Singh.pdf)、[論文](https://arxiv.org/pdf/1602.04938.pdf)
-   [EXPLAINABLE MACHINE LEARNING 簡報](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/xai_v4.pdf)
-   [SHAP（SHApley Additive exPlanations）](https://github.com/slundberg/shap)
-   [LIME](https://github.com/marcotcr/lime)
