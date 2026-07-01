# 电影评论分类

一个基于 NLTK 电影评论语料的文本分类实验，对比朴素贝叶斯与线性 SVM 在情感分类任务中的结果。

## 功能

- 自动检查并准备 `movie_reviews` 语料
- 对评论文本进行清洗、筛选和词干化
- 使用 `TF-IDF` 构造文本特征
- 训练朴素贝叶斯模型与线性 SVM 模型
- 输出准确率、精确率、召回率和 F1
- 输出分类报告与混淆矩阵
- 生成模型对比图和特征重要性图

## 技术栈

- Python
- NLTK
- scikit-learn
- matplotlib
- NumPy

## 开发环境

- Python 3.x

## 项目结构

- 主脚本：`sentiment_classifier_main.py`
- 早期版本：`archive/sentiment_classifier_early.py`

## 核心实现

### 文本预处理

脚本会对原始评论执行以下处理：

- 转小写
- 移除 HTML 标记
- 用正则提取字母型 token
- 去停用词
- 使用 `PorterStemmer` 做词干化

### 特征提取

文本特征由 `TfidfVectorizer` 生成，后续送入分类模型进行训练和预测。

### 模型训练

代码包含两类模型：

- `MultinomialNB`
- `LinearSVC`

训练后会对测试集进行预测，并计算多项分类指标。

### 结果输出

脚本会输出：

- Accuracy
- Precision
- Recall
- F1
- Classification Report
- Confusion Matrix

同时生成模型对比图、混淆矩阵图和特征重要性图。

## 如何运行

运行主版本：

```powershell
python sentiment_classifier_main.py
```

如需查看早期版本：

```powershell
python archive/sentiment_classifier_early.py
```

如果本地缺少语料，脚本会在项目目录下的 `nltk_data` 中准备所需资源。

## 项目展示

运行后会生成模型对比图、混淆矩阵图和特征重要性图，可直接用于实验结果展示。

## 后续优化方向

- 统一主版本和早期版本中的公共流程
- 增加固定随机种子与实验配置说明
- 补充更多模型对比

## AI 参与说明

项目开发过程中使用了 AI 作为辅助工具，主要用于实验思路梳理、代码调试、库文档查阅和问题分析；实际运行、结果检查、图像生成和代码调整由本人完成。
