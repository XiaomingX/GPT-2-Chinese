**状态：** 归档（代码按现状提供，预计不会更新）

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

### 与我们合作

如果你正在开展关于GPT-2的有趣研究或应用工作，请[告知我们](mailto:languagequestions@openai.com)！我们尤其有兴趣听取并可能与以下研究方向的人员合作：
- 潜在的恶意用例及其防御措施（例如合成文本的可检测性）
- 模型中存在的问题内容（如偏见）的程度以及有效的缓解措施

## 开发

参见[DEVELOPERS.md](./DEVELOPERS.md)

## 贡献者

参见[CONTRIBUTORS.md](./CONTRIBUTORS.md)

## 引用

请使用以下bibtex条目：
```
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```

## 未来工作

我们可能会发布在各种基准上评估模型的代码。

我们仍在考虑发布更大的模型。

## 许可证

[修改后的MIT许可证](./LICENSE)