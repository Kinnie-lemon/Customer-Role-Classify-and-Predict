# Customer Role Classify and Predict
Based on labeled text set to accomplish a multi-class problem



![image](https://user-images.githubusercontent.com/54998666/111857711-54f9d880-896e-11eb-91c3-56c534d28fd8.png)

****
## 1.Preprocess
### a)导入各类Role数据并打标签，存入Label列
### b)Data clean
  - i.Business scope
  1. split words
  2. remove stop words and signposting language
  3. keep Chinese characters only
  4. remove brackets（CH&EN）and words wrapped
  - ii.Company name：where extract industry key words from
    1. 去地名、去公司末尾后分词，取最后一个词。 如xx科技有限公司，科技被提取出来
    2. 如果正好是两个或三个字，就直接通过。否则，到step3
    3. 如果四个及以上字，就提取最后两个字

### c)TF-IDF
#### i.object：word2vec
#### ii.variable table
|Variables	|Description|
|:--|:--|
|Words_list|	unique words list for all data
|Document|	Vocabulary（词库）
|Document_train|	World list for train set 
|Sparse_result|	Tf-idf for all items
|tfidf_model_train|	Tf-idf for training items
|X|	Tf-idf in shape(samples,training words)
|Df_samples|	Inner joined data of full and categorical data sets,; Role categories have been encoded

#### iii.模型逻辑
1. based on all words to vocabulary for TF-IDF Model
2. extract training-set involved words进行模型训练以提高运行速度
## 2.Models
实际使用时预处理流程相同，模型部分直接调用.pkl文件进行预测，读取的数据如下：
|Variables	|Description|
|:--|:--|
|name_EC| 	Key words list of current Role
|name_EC_excl|	excluded Key words list of current Role
|clf_SI|	Model Layer 1 
|clf_SI_stack|	Model Layer 2

### a)Preprocess
- i.extract key words form company names
- ii.判断每个公司名称中是否包含name_EC或name_EC_excl中的关键词： 1/0
- iii.平衡模型：从各个Roles中取50个样本，根据是否是当前讨论的Role进行二分类（正负集）
- iv.经营范围TF-IDF array：全正负集&当前Role set
- v.注意：PaB经营范围关键词增加['低压','开关','配电']，OEM经营范围关键词增加 ['机器人','铸造','智能']
- vi.注意：所用的tfidf模型，去除地区所用的geo_list，和industry_list（地理位置及行业词库）可直接导入.pkl文件
### b)Model Stacking
- i.SVM（只考虑经营范围TF-IDF）
  1. K折交叉验证（K=10）
  2. 线性核SVM，惩罚参数C=5
- ii.SVM（stack模型1结果&添加公司名称feature）
  1. Input_columns = model1_output，EC_Name_classify_output, EC_Name_exclude_classify_output
  2. K折交叉验证（K=10）
  3. rbf核SVM，惩罚参数C=5
- iii.SVM（stack模型2结果，不用交叉验证）
  1. 线性核SVM，惩罚参数C=5
    - a)Layer 1 ：经营范围TF-IDF
    - b)Layer 2 ：Layer 1 output & EC_Name_classify_output & EC_Name_exclude_classify_output（理解为name_feature）
    - c)输出：
|Variables	|Description|
|:--|:--|
|KW_incl|	是否包含当前Role的key word
|KW_excl|	是否包含非当前Role的key word
|Prediction|	Layer 1 分类结果：0/1
|Prediction_2|	Layer 2 分类结果：0/1
|'Prediction_2_Prob_other', 'Prediction_2_Prob1_EC‘	Layer2| 分为当前Role和Others各自的概率
|Final_Pred|	如果key word没有被成功分类，则标记为1；否则取Layer2分类结果


## 3.Result Output
### a)目标：如果五类的预测概率没有一类大于0.5，就认为是其他，否则选取概率最高的那一类
### b)合并模型预测结果
- i.取每个模型的二分类结果及分类预测概率两列（上述prediction2及Prediction_2_Prob1_EC列
- ii.将5个模型结果合并
- iii.其中EC多了一列df_pred_ec_prob_hidden，逻辑为：属于ec（二分类=1）则ec预测概率+0.2，否则不变
- iv.输出列：['EC', 'PaB', 'OEM', 'SI', 'Dist', 'EC_prob', 'PaB_prob', 'OEM_prob', 'SI_prob', 'Dist_prob', 'EC_hidd']
### c)预测结果筛选
- i.返回五个中概率的最大值为最终预测结果
- ii.去掉预测结果字符串的后五位'_prob'
- iii.过滤出Others类
  1. 如果五类的预测概率没有一类大于0.5，就认为是Others，否则选取概率最高的那一类
  2. 更新数据，drop ['EC_hidd', 'is_others']列
## 5.其他
### a)各分类关键词提取
- i.获取train_corpus的TF-IDF.array
- ii.多分类逻辑回归获取每个词的提取系数
