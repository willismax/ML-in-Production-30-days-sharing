# [Day 06 : 什麼是 MLOps](https://ithelp.ithome.com.tw/articles/10260304)

###### tags: `MLOps`
[![](https://d1dwq032kyr03c.cloudfront.net/images/ironman_sticker/13/ai-and-data.png?sticker "第 13 屆鐵人賽鍊成")第 13 屆鐵人賽鍊成](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)
[![](https://img.shields.io/badge/iThome%E9%90%B5%E4%BA%BA%E8%B3%BD2021-%E5%A8%81%E5%88%A9%E6%96%AF-blue)](https://ithelp.ithome.com.tw/articles/10260304)

各種商務情境都在思考如何融入 AI 提供更適切的智慧化服務，在[Day 04 : 以資料為中心的人工智慧 Data-Centric AI](https://ithelp.ithome.com.tw/articles/10259708) 介紹透過關注資料為中心的 AI 焦點，更結構化的思考用於生產中的 ML 系統， MLOps 是種系統化思考的精神與落實的方針，也是發展中的學門，在這未定容許思辨的時刻，以下分為微軟、Google、持續交付基金會(CDF)的定義與說明:

MLOps 定義
--------

### Google

-   Google 在 [An introduction to MLOps on Google Cloud](https://www.youtube.com/watch?v=6gdrwFMaEZ0) 影片中簡要說明:
    
    > MLOps 是種工程文化與實踐，旨在 ML 系統開發與 ML 系統操作。
    

### 微軟

-   微軟在 [MLOps on Azure](https://github.com/microsoft/MLOps) 介紹說明:
    
    > -   MLOps 是使資料科學家和應用程式開發人員能夠幫助將機械學習模型投入生產。
    > -   MLOps 使您能夠追蹤、版控、稽核、認證、重複使用 ML 生命週期中的每項資產及簡化管理生命週期。
    

### 持續交付基金會CDF

-   孵育出 [Jenkins](https://www.jenkins.io/) 開源專案與社群，由 Linux 基金會成立的「[持續交付基金會(CDF)](https://cd.foundation/)」也意識到ML佈署到現實情境中的問題，於2020年2月宣布成立 [MLOps SIG (CDF Special Interest Group - MLOps)](https://cd.foundation/blog/2020/02/11/announcing-the-cd-foundation-mlops-sig/)，依據 CDF [MLOps Roadmap 2021(草案)](https://github.com/cdfoundation/sig-mlops/blob/master/roadmap/2021/MLOpsRoadmap2021.md)，將 MLOps 定義為:
    
    > -   MLOps 為 "DevOps" 方法論的擴展，並將機器學習和數據科學資產作為 DevOps 生態中的一等公民納入其中。
    > -   MLOps 應被視為一種實踐，以與所有其他技術和非技術要素統一的方式持續管理產品的 ML 方面，以成功將這些產品商業化，並在市場上具有最大的可行性。這也包括 DataOps，因為沒有完整、一致、語義有效、正確、及時和無偏見的數據的機器學習是有問題的，或者導致可能加劇內置偏見的有缺陷的解決方案。
    
    -   ![](https://i.imgur.com/KXOjfYp.png)
        -   圖片來源: [CDF 的 MLOps 介紹](https://cd.foundation/blog/2020/05/29/mlops-an-introduction/)
-   MLOps 是與不是
    
    -   MLOps 旨在您的 ML 產品服務開發、佈署及維運過程中，開發階段能持續整合、佈署能實現持續佈署，更藉由 ML Pipeline 縮短溝通成本，讓資料科學家能自動化並在生產系統中獲得寶貴的見解，讓營運團隊提供可再現性、可見性及託管服務和計算支援。
    -   另外， [CDF MLOps Roadmap 2021(草案)](https://github.com/cdfoundation/sig-mlops/blob/master/roadmap/2021/MLOpsRoadmap2021.md#what-is-mlops-not)，也提出**MLOps 並不是"將 Jupyter Notebooks 放入生產環境"** (這句話莫名莞爾)。  
        ![/images/emoticon/emoticon07.gif](https://ithelp.ithome.com.tw/images/emoticon/emoticon07.gif)

MLOps 與 DevOps 關係與區隔
--------------------

-   MLOps 源自於資料科學團隊與營運團隊2者之間的彼此合作與需求，如同 DevOps 期待整合開發與維運實現敏捷開發的過程。
-   MLOps 與 DevOps 目標一致，縮短系統開發週期，確保持續開發高品質的軟體服務， CI/CD 並在生產中維護，但 ML 有它獨特需要注意的不同需求， 版本控制是為了追蹤與捕捉資料與世界的變化，不僅僅是紀錄程式本身。
-   除了 ML code，諸多在 ML 專案實現、維運過程中的流程組合構成 MLOps 。

MLOps 與 Data-centric AI
-----------------------

-   MLOps 讓處理資料更系統化，不再將資料處理任務停留在是骯髒、臨時起意、靠經驗的印象，而是採更系統化的方式看待整個工作流程，符合以系統化看待以資料為中心的 AI 。
-   在 [A Chat with Andrew on MLOps: From Model-centric to Data-centric AI](https://www.youtube.com/watch?v=06-AZXmwHjo) 有吳恩達的說明及討論，這也是他第一次公開說明 Data-centric AI，在系列文 [Day 04](https://ithelp.ithome.com.tw/articles/10259708) 有些說明可參考。

小結
--

-   MLOps 是持續發展的新興學門，本日試圖摘述不同 MLOps 的定義論點，筆者傾向能一句話講清楚的方式: **MLOps 是一種工程文化與實踐，旨在 ML 系統開發與 ML 系統操作**。可以讓更多人認識 MLOps 。
-   MLOps 的實現可以讓用於生產中的 ML 服務，具有多管道安排的能力、管理 ML 產品生命週期的能力、擴展 ML 服務的能力、追蹤 ML 系統健康的能力、CI/CD、CT、ML 模型治理的能力。

參考
--

-   [An introduction to MLOps on Google Cloud](https://www.youtube.com/watch?v=6gdrwFMaEZ0)
-   [MLOps on Azure](https://github.com/microsoft/MLOps)
-   [Announcing the CD Foundation MLOps SIG](https://ithelp.ithome.com.tw/articles/(https://cd.foundation/blog/2020/02/11/announcing-the-cd-foundation-mlops-sig/))
-   [MLOps Roadmap 2021 - DRAFT VERSION](https://github.com/cdfoundation/sig-mlops/blob/master/roadmap/2021/MLOpsRoadmap2021.md)
-   [A Chat with Andrew on MLOps: From Model-centric to Data-centric AI](https://www.youtube.com/watch?v=06-AZXmwHjo)

