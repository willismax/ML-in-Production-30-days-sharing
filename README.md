# ML-in-Production-30-days-sharing

2021 IT 邦幫忙鐵人賽 「從 AI 落地談 MLOps 系列」系列文

[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)

[第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)

## 從 AI 落地談 MLOps
行百哩路半九十，即便各種先進的AI模型如雨後春筍般化為現實，AI成功落地佈署至商業情境仍是困難重重，營運中的商業服務如何調整其ML算法，讓服務經得起時間及使用者的考驗，必須反覆推敲範疇、資料、模型及佈署的問題，進一步而言，有沒有關於AI落地、ML in Production、MLOps的解決方案?

既然佈署機械學習ML的工程實務逐漸歸納並越受重視，也多虧軟體工程DevOps精神興起與CI/CD實務越漸普及，本系列將梳理期待透過鐵人賽將系列知識做個梳理，拋磚引玉，協助銜接ML商務落地的一哩路，以及對AI有志趣者，補充除了AI建模以外需要關注的面向。


### 談觀念:
- 此階段闡述 MLOps 精神與旨趣，呼應「以資料為中心的人工智慧」的想法，將用於生產的機械學習應有做為，以機械學習系統的生命週期解構從範疇、資料、建模到佈署應注意的人與事。
    - [Day 01 : 這系列文在做什麼-緣起](https://ithelp.ithome.com.tw/articles/10258837)
    - [Day 02 : 用於生產的機械學習 ML in Production](https://ithelp.ithome.com.tw/articles/10258861)
    - [Day 03 : ML in Production 的挑戰](https://ithelp.ithome.com.tw/articles/10259314)
    - [Day 04 : 以資料為中心的人工智慧 Data - Centric AI](https://ithelp.ithome.com.tw/articles/10259708)
    - [Day 05 : ML 專案生命週期](https://ithelp.ithome.com.tw/articles/10259989)
    - [Day 06 : 什麼是 MLOps](https://ithelp.ithome.com.tw/articles/10260304)
    - [Day 07 : MLOps 的挑戰與技術要求](https://ithelp.ithome.com.tw/articles/10260599)
    - [Day 08 : ML 工程師職責與分工](https://ithelp.ithome.com.tw/articles/10260962)
    - [Day 30 : 綜合整理 MLOps Level 0 ~ 2](https://ithelp.ithome.com.tw/articles/10274317)
### 談實踐
技術介紹隨著機械學習生命週期推進，系列主軸希望介紹由 Google 開源的 TensorFlow Extended (TFX) 做為用於生產的機械學習框架，也以ML系統生命週期介紹實踐與模型優化方法。
- 範疇領域 Scope
    - [Day 09 : 用於生產的機械學習 - 定義範疇 Scope](https://ithelp.ithome.com.tw/articles/10261352)
- 資料領域 Data
    - [Day 10 : 用於生產的機械學習 - Data Define 與建立基準](https://ithelp.ithome.com.tw/articles/10261664)
    - [Day 11 : 用於生產的機械學習 - Data Labeling 資料標註](https://ithelp.ithome.com.tw/articles/10262021)
    - [Day 12 : 弱監督式標註資料 Snorkel (spam 入門篇)](https://ithelp.ithome.com.tw/articles/10262325) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/willismax/ML-in-Production-30-days-sharing/blob/main/notebook/12.%E5%85%A5%E9%96%80_Snorkel_.ipynb)
    - [Day 13 : 弱監督式標註資料 Snorkel (視覺關係偵測篇)](https://ithelp.ithome.com.tw/articles/10262699) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/willismax/ML-in-Production-30-days-sharing/blob/main/notebook/13.Snorkel_%E8%A6%96%E8%A6%BA%E9%97%9C%E4%BF%82%E6%AA%A2%E6%B8%ACVRD%E7%AF%84%E4%BE%8B.ipynb)
    - [Day 14 : 資料驗證 TensorFlow Data Validation (TFDV)](https://ithelp.ithome.com.tw/articles/10263091) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/data_validation/tfdv_basic.ipynb) 
    - [Day 15 : 特徵工程 tf.Tramsform 介紹](https://ithelp.ithome.com.tw/articles/10263595)
    - [Day 16 : 特徵工程 tf.Tramsform 實作](https://ithelp.ithome.com.tw/articles/10264084) [官方 ![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/tensorflow/tfx/blob/master/docs/tutorials/transform/simple.ipynb)
    - [Day 17 : 用於生產的機械學習 - 特徵選擇 Feature Selection](https://ithelp.ithome.com.tw/articles/10264846) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/willismax/ML-in-Production-30-days-sharing/blob/main/notebook/17.%E7%89%B9%E5%BE%B5%E9%81%B8%E6%93%87_Deta_Selection_%E9%90%B5%E4%BA%BA%E8%B3%BD%E7%A4%BA%E7%AF%84_.ipynb)
- 建模領域 Modeling
    - [Day 18 : 深度學習(神經網絡)自動調參術 - KerasTuner](https://ithelp.ithome.com.tw/articles/10265801) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/willismax/ML-in-Production-30-days-sharing/blob/main/notebook/18.Keras_Tuner_%E9%90%B5%E4%BA%BA%E8%B3%BD%E5%88%86%E4%BA%AB.ipynb)
    - [Day 19 : 深度學習(神經網絡)自動建模術 - AutoMLs](https://ithelp.ithome.com.tw/articles/10266499) [ ![](https://i.imgur.com/pQnQ4tG.png)]( https://colab.research.google.com/github/keras-team/autokeras/blob/master/docs/ipynb/image_classification.ipynb)
    - [Day 20 : 模型優化 - 訓練後量化 Post Training Quantization](https://ithelp.ithome.com.tw/articles/10267328)  [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/willismax/ML-in-Production-30-days-sharing/blob/main/notebook/20.%E8%A8%93%E7%B7%B4%E5%BE%8C%E9%87%8F%E5%8C%96_TensorFolw_Lite_Quantization_%E9%90%B5%E4%BA%BA%E8%B3%BD%E7%A4%BA%E7%AF%84.ipynb)
    - [Day 21 : 模型優化 - 剪枝 Pruning](https://ithelp.ithome.com.tw/articles/10268124)  [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/willismax/ML-in-Production-30-days-sharing/blob/main/notebook/21.%E6%A8%A1%E5%9E%8B%E5%84%AA%E5%8C%96_%E5%89%AA%E6%9E%9D_Pruning_.ipynb)
    - [Day 22 : 模型優化 - 知識蒸餾 Knowledge Distillation](https://ithelp.ithome.com.tw/articles/10268783) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/willismax/ML-in-Production-30-days-sharing/blob/main/notebook/22.%E7%9F%A5%E8%AD%98%E8%92%B8%E9%A4%BE_%E9%90%B5%E4%BA%BA%E8%B3%BD%E7%A4%BA%E7%AF%84.ipynb)
    - [Day 23 : 模型分析 TensorFlow Model Analysis (TFMA)](https://ithelp.ithome.com.tw/articles/10269467)[![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/tensorflow/tfx/blob/master/docs/tutorials/model_analysis/tfma_basic.ipynb#scrollTo=SA2E343NAMRF)
    - [Day 24 : 負責任的 AI - Responsible AI (RAI)](https://ithelp.ithome.com.tw/articles/10270241)
    - [Day 25 : 可解釋的 AI - Explain AI (XAI)](https://ithelp.ithome.com.tw/articles/10270902)
    - [Day 26 : 公平指標與實作 Fairness Indicators](https://ithelp.ithome.com.tw/articles/10271626) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_TFCO_CelebA_Case_Study.ipynb#scrollTo=GRIjYftvuc7b)
- 部署領域 Deploying
    - [Day 27 : 使用 TensorFlow Serving 部署 REST API](https://ithelp.ithome.com.tw/articles/10272257) [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/github/willismax/ML-in-Production-30-days-sharing/blob/main/notebook/27.TensorFlow_Serving_REST_API_%E9%90%B5%E4%BA%BA%E8%B3%BD%E7%A4%BA%E7%AF%84.ipynb)
    - [Day 28 : 用於生產的機械學習 TensorFlow Extended (TFX) 介紹](https://ithelp.ithome.com.tw/articles/10272958)
    - [Day 29 : 用於生產的 TensorFlow Extended (TFX) 實作](https://ithelp.ithome.com.tw/articles/10273652) [ ![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1o4lRoAdpPkfCL6WV3X6JwXK5C27itbI6?usp=drive_fs)
