# [Day 13 : 弱監督式標註資料 Snorkel (視覺關係偵測篇)](https://ithelp.ithome.com.tw/articles/10262699)


###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10262699)

-   接續 [Day 12](https://ithelp.ithome.com.tw/articles/10262325)的弱監督式 Snorkel 範例，今天再花點時間示範用 Snorkel 標註影像資料。
-   Snorkel 透過簡易廣泛的程式撰寫判斷邏輯後，交由生成對抗網路產生分類結果，分類的結果效果不差，而且不用手動標註。之後也會介紹 AutoML 等工具，在此之前我們來透過 Snorkel 官方範例了解如何進行。
-   Colab 實作範例 [![](https://i.imgur.com/pQnQ4tG.png)](https://colab.research.google.com/drive/1WsbfDk9r_g9_75CuQNmkvn3GzDhExmjc)。

Visual Relationship Detection, VRD 視覺關係偵測
-----------------------------------------

-   VRD 說明:
    -   通常圖片內容物都有物體之間的關聯性，定義描述為為`a subject <predictate> object` 。例如， `person riding bicycle` ， “person” 和 “bicycle” 分別是主詞和受詞， “riding” 是關係動詞。
    -   此範例源自 [snorkel-tutorials](https://github.com/snorkel-team/snorkel-tutorials/blob/master/visual_relation/visual_relation_tutorial.ipynb)，目的為對[視覺關係檢測 (VRD) 數據集](https://cs.stanford.edu/people/ranjaykrishna/vrd/)進行操作，專注於圖片內物件之間的關係分類任務。
    -   以下圖示紅色框代表主題，而綠色框代表對象。該謂詞（如踢）表示什麼關係連接主體和客體。

![](https://camo.githubusercontent.com/a4dd89f1ae5fbffbc4349daf6b32cbaf85fb7e02/68747470733a2f2f63732e7374616e666f72642e6564752f70656f706c652f72616e6a61796b726973686e612f7672642f646174617365742e706e67)

### 0\. 設定環境

-   筆者有調整為 Colab 可以執行的程式，不過範例主程式裡執行的`model.py`遇到`pandas.DataFrame.as_matrix()`是舊的語法，也提供修正方式`pandas.DataFrame.values`。
-   當然您也可以如官網範例指定舊版的 pandas ，相關設定不贅述。

### 1\. 加載數據

-   範例將訓練集、有效集和測試集加載為 DataFrame。
-   數據集的採樣版本在訓練集、開發集和測試集上使用相同的 26 個數據。此設置旨在快速演示 Snorkel 如何處理此任務，而非演示性能。
-   ![](https://i.imgur.com/HCvIjtN.png)

### 2\. 編寫 Labeling Functions (LFs)

-   我們現在編寫標記函數來檢測邊界框對之間存在什麼關係。為此，我們可以將各種直覺編碼到標記函數中：
    
-   分類直覺：關於這些關係中通常涉及的主詞和受詞類別的知識（例如，person通常是謂詞 RIDE 和的主詞 CARRY）
    
-   空間直覺：關於主詞和受詞的相對位置的知識（例如，主詞通常高於動詞的受詞RIDE）
    
    ```python
    RIDE = 0
    CARRY = 1
    OTHER = 2
    ABSTAIN = -1
    
    ```
    
-   我們從編碼分類直覺的標記函數開始：我們使用關於共同的主題-客體類別對的知識 RIDE，CARRY 以及關於哪些主題或客體不太可能涉及這兩種關係的知識。
    
    ```python
    from snorkel.labeling import labeling_function
    
    # Category-based LFs
    @labeling_function()
    def lf_ride_object(x):
        if x.subject_category == "person":
            if x.object_category in [
                "bike",
                "snowboard",
                "motorcycle",
                "horse",
                "bus",
                "truck",
                "elephant",
            ]:
                return RIDE
        return ABSTAIN
    
    
    @labeling_function()
    def lf_carry_object(x):
        if x.subject_category == "person":
            if x.object_category in ["bag", "surfboard", "skis"]:
                return CARRY
        return ABSTAIN
    
    
    @labeling_function()
    def lf_carry_subject(x):
        if x.object_category == "person":
            if x.subject_category in ["chair", "bike", "snowboard", "motorcycle", "horse"]:
                return CARRY
        return ABSTAIN
    
    
    @labeling_function()
    def lf_not_person(x):
        if x.subject_category != "person":
            return OTHER
        return ABSTAIN
    
    ```
    
-   現在編碼空間直覺，其中包括測量邊界框之間的距離並比較它們的相對區域。
    
    ```python
    YMIN = 0
    YMAX = 1
    XMIN = 2
    XMAX = 3
    
    ```
    
    ```python
    import numpy as np
    
    # Distance-based LFs
    @labeling_function()
    def lf_ydist(x):
        if x.subject_bbox[XMAX] < x.object_bbox[XMAX]:
            return OTHER
        return ABSTAIN
    
    
    @labeling_function()
    def lf_dist(x):
        if np.linalg.norm(np.array(x.subject_bbox) - np.array(x.object_bbox)) <= 1000:
            return OTHER
        return ABSTAIN
    
    
    def area(bbox):
        return (bbox[YMAX] - bbox[YMIN]) * (bbox[XMAX] - bbox[XMIN])
    
    
    # Size-based LF
    @labeling_function()
    def lf_area(x):
        if area(x.subject_bbox) / area(x.object_bbox) <= 0.5:
            return OTHER
        return ABSTAIN
    
    ```
    
-   標記函數具有不同的經驗準確性和覆蓋範圍。由於我們選擇的關係中的類別不平衡，標記 OTHER 的標記函數比RIDE或CARRY的標記函數具有更高的覆蓋率。這也反映了數據集中類的分佈。
    

### 3\. 訓練標籤模型

-   訓練 `LabelModel` 來為未標記的訓練集分配訓練標籤。
    
    ```python
    from snorkel.labeling.model import LabelModel
    
    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(
        L_train, 
        seed=123, 
        lr=0.01, 
        log_freq=10, 
        n_epochs=100
        )
    
    ```
    

### 4\. 訓練分類器

-   現在，您可以使用這些訓練標籤來訓練任何標準判別模型，例如現成的 [ResNet](https://github.com/KaimingHe/deep-residual-networks)，它應該學會在我們開發的 LF 之外進行泛化。
    
    ```python
    from snorkel.classification import DictDataLoader
    from model import SceneGraphDataset, create_model
    
    df_train["labels"] = label_model.predict(L_train)
    
    if sample:
        TRAIN_DIR = "data/VRD/sg_dataset/samples"
    else:
        TRAIN_DIR = "data/VRD/sg_dataset/sg_train_images"
    
    dl_train = DictDataLoader(
        SceneGraphDataset("train_dataset", "train", TRAIN_DIR, df_train),
        batch_size=16,
        shuffle=True,
    )
    
    dl_valid = DictDataLoader(
        SceneGraphDataset("valid_dataset", "valid", TRAIN_DIR, df_valid),
        batch_size=16,
        shuffle=False,
    )
    
    ```
    
-   定義模型架構。
    
    ```python
    import torchvision.models as models
    
    # initialize pretrained feature extractor
    cnn = models.resnet18(pretrained=True)
    model = create_model(cnn)
    
    ```
    
    -   訓練與評估模型
    
    ```python
    from snorkel.classification import Trainer
    
    trainer = Trainer(
        n_epochs=1,  # increase for improved performance
        lr=1e-3,
        checkpointing=True,
        checkpointer_config={"checkpoint_dir": "checkpoint"},
    )
    trainer.fit(model, [dl_train])
    
    ```
    
    ```python
    model.score([dl_valid])
    # {'visual_relation_task/valid_dataset/valid/f1_micro':
    #  0.34615384615384615}
    
    ```
    
-   我們已經成功訓練了一個視覺關係檢測模型！使用關於視覺關係中的對像如何相互作用的分類和空間直覺，我們能夠在多類分類設置中為 VRD 數據集中的對像對分配高質量的訓練標籤。
    
-   有關 Snorkel 如何用於視覺關係任務的更多信息，請參閱該團隊 [ICCV 2019 論文](https://arxiv.org/abs/1904.11622)。
    

小結
--

-   這一篇是筆者想確認如何用弱監督的方式完成影像標註，Snorkel 確實做到了，但可惜的是相依模組版本比較舊，在 Colab 實現需要些調整，筆者調整後就分享給有興趣的人。
-   現在無監督式學習興起，但如果有需要退而求其次自己寫條件時，Snorkel 應可幫助到您。

參考
--

-   [snorkel-tutorials](https://github.com/snorkel-team/snorkel-tutorials/blob/master/visual_relation/visual_relation_tutorial.ipynb)
