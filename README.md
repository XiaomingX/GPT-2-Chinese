# GPT-2

来自论文《语言模型是无监督多任务学习者》（["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)）的代码和模型。

你可以通过我们的[原始博客文章](https://openai.com/research/better-language-models/)、[6个月后续文章](https://openai.com/blog/gpt-2-6-month-follow-up/)和[最终文章](https://www.openai.com/blog/gpt-2-1-5b-release/)了解GPT-2及其分阶段发布过程。

我们还[发布了一个数据集](https://github.com/openai/gpt-2-output-dataset)，供研究人员研究其行为。

<sup>*</sup> *注意：由于一个错误，我们最初的参数数量统计有误（在之前的博客文章和论文中）。因此，你可能看到过小型模型被称为117M，中型模型被称为345M。*

## 使用方法

本仓库旨在作为研究人员和工程师实验GPT-2的起点。

基础信息请参见我们的[模型卡片](./model_card.md)。

### 一些注意事项

- GPT-2模型的鲁棒性和最坏情况行为尚未被充分理解。与任何机器学习模型一样，需针对你的使用场景仔细评估GPT-2，尤其是在未经过微调的情况下，或在可靠性至关重要的安全关键型应用中使用时。
- 训练GPT-2模型所用的数据集包含许多带有偏见和事实错误的文本，因此GPT-2模型也可能存在偏见和不准确之处。
- 为避免样本被误认为是人类撰写的，建议在广泛传播前明确将样本标记为合成内容。我们的模型在细微之处常常不连贯或不准确，这需要人类仔细阅读才能发现。
