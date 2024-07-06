# 博士论文

中文标题：高能效近似乘法器设计及综合研究

英文标题：High-Efficient Approximate Multiplier Design and Synthesis

master分支：开源， reviewerC分支：盲审及终稿。

## 明审

#### 评阅意见

论文规范性良好，概念较为清晰。（意思就是规范性写得稀烂，概念之间串不起来）

#### 修改建议

1. 建议统一所有图的格式，包括图中所有字的大小应统一，且尽量与文本中的字号大小一致，将较小的图放大使其更为清晰（例如图5-9），将较大的图缩小（如图5-6）等。同时，建议进一步规范论文格式，如统一行间距等。（`原图5-9变为现图2-35、2-36并适当放大，原图5-6变为现图2-32并适当缩小`）
2. 虽然本文围绕近似电路设计展开，但三个部分相对比较独立。建议进一步阐述这三个部分之间的关系，使其成为一个有机整体。（`与reviewerC的建议一致`）

#### 修改后当面建议 6月26号

1. 缩短摘要，中文在一页纸以内，英文在一页半；（`最后中文一页半，英文两页多一点`）
2. 组织结构图改一下，箭头太大，且第一章第二章不需要；（`最后改了组织结构图，留下了第一章和第二章`）
3. 章引言标题去掉，5.1内容用文心一言润色，且明确告知文心一言要往“逻辑性更强”这一方向去优化；（`没去引言标题，5.1 引言手动润色了一下`）
4. 第二章开头，引入引言，否则有点突兀。部分积的生成、累加、最终相加三个步骤，分点itemize；（`√`）
5. 过长的段要分段；
6. 一个句号不能超过一行半；
7. 所有图片的文字，尽量统一成中文（除非真的要用英文），大小尽量和正文一致（除了图中不重要的文字）；
8. 如果某一个标题的内容太少，就把标题去掉，用文字来过度，如1.3 国内外研究现状、4.3的各小标题；（`1.3 已改`）
9. 5.2.1 标题有点像国内外研究现状，可以去掉该标题，改为文字过渡；


## reviewerC

#### 评阅意见

论文逻辑性差，结构欠合理，写作需提升。

#### 修改建议

1. 写作有待全面提升。比如摘要，在叙述各部分工作之前应该有一段对该论文的三个工作做总结性描述，在每一个研究内容描述的的第一句也应该对该部分工作最关键的信息用一句话进行描述，而目前摘要的写法读者读完之后一头雾水，不知道该论文主要做了些什么工作，有哪些创新。再比如各章都写了研究背景与现状及研究动机，这些内容应该全部放在绪论章，并设法在摘要、绪论的本文主要工作和组织结构及章引言中阐明各部分内容之间的逻辑关系，使得各部分内容串起来成为一个有机的整体，而不是三个部分的简单拼凑，三个部分的简单拼凑是不能构成一篇博士论文的。

## 修改说明

（首行缩进2字符：简况表是空8个空格，复审表是空4个。专家只会看到简况表，因此空8个为宜）

尊敬的专家评审：

您好！非常感谢您百忙之中审阅我的毕业论文，并给出了修改建议。在收到评阅意见后，笔者与指导教师及同学开展了多次交流，认为您的评阅意见正确且合理。根据您指出的论文不足之处，我进行了修改，并在您给出的修改意见的基础上，对论文做了进一步修改。下面从这两个方面展开说明：

一、您的建议

- 评审意见1：摘要在叙述各部分工作之前应该有一段对该论文的三个工作做总结性描述，在每一个研究内容描述的的第一句也应该对该部分工作最关键的信息用一句话进行描述。

修改说明：

感谢评审专家提出的宝贵意见。在深刻检讨论文组织结构和书写形式后，论文对摘要内容做了重新的梳理和书写，按照总-分-总的形式，对研究背景、核心研究问题、子问题的研究成果进行了文字上的凝练。具体修改内容如下：

(1) 针对摘要缺乏对三个工作总结性描述的问题，修改后的论文的摘要部分在叙述每个工作的具体研究内容及成果之前，简要介绍了论文的三个工作及之间的联系：由于现有近似乘法器相关设计缺乏对数据统计学特性的利用，并且存在手工设计效率低以及在不同架构上硬件收益不匹配（主要指ASIC和FPGA）的缺点。 因此本文提出了两种开源的自动化近似乘法器设计方法，能够高效地生成不同精度的适用于 ASIC 和 FPGA 的高能效近似乘法器。结合得到的近似乘法器，本文基于强化学习方法利用近似逻辑综合技术研究了不同近似乘法器对大规模电路（以DNN硬件加速器为例）整体PPA带来的影响。

(2) 针对摘要中每个研究内容缺乏一句话描述关键信息的问题，修改后的论文的摘要部分在叙述每个研究工作具体内容的开头，用一句话提炼了该工作的关键信息，归纳了创新点，以使每个部分条理清晰，便于理解。具体包括：(a) 面向 ASIC，本文基于数据分布和输入极性提出了一种开源的自动化近似乘法器设计方法，该方法能够高效地生成适应于特定应用的高性能ASIC近似乘法器，提高应用的运算效率。(b) 由于ASIC和FPGA底层架构不同，ASIC近似乘法器通常无法在FPGA上取得相同比例的硬件性能提升。因此面向FPGA应用，本文基于贝叶斯优化提出了一种开源的自动化近似乘法器生成方法，避免了手工修改查找表编码方法效率较低的问题。(c) 结合前面两个工作，面向大规模电路，本文基于强化学习方法和得到的近似乘法器进行了近似逻辑综合研究。

- 评审意见2: 各章都写了研究背景与现状及研究动机，这些内容应该全部放在绪论章。

修改说明：

感谢评审专家的意见。针对研究背景与现状及研究动机的内容及位置不合理这一问题，已根据意见优化了绪论章的内容及结构，主要修改内容如下：

(1) 将原第3章、第4章、第5章的研究背景、现状及研究动机拆分、合并到了“第1章 绪论”和第2章，并将第2章的标题改为“乘法器与逻辑综合技术基础”以便和章节内容相统一。

(2) 将原“1.1 研究背景与意义”拆分为“1.1 研究背景”与“1.2 研究意义”两个部分，将原1.1涉及到近似计算优势的内容放在1.2中，并对1.2增加了有关研究近似乘法器的意义、近似乘法器与逻辑综合之间的联系等内容。

(3) 增加“1.3 国内外研究现状”，从近似乘法器、逻辑综合以及近似逻辑综合三个方面递进地描述有关高能效近似乘法器设计及综合的国内外研究现状，分别包括：近似乘法器在功能近似层面研究的3个阶段，逻辑综合的分类、研究内容，近似逻辑综合研究现状。

(4) 增加1.4 研究动机，重点阐述了本文提出两种自动化近似乘法器设计方法的原因，以及为何要把得到的近似乘法器和逻辑综合结合起来进行近似逻辑综合的研究。对ASIC近似乘法器来讲，已有的近似乘法器设计方法通常假设输入是均匀分布的，且没有考虑对称性对精度产生的影响，因此本文提出了基于数据分布和输入极性的设计方法。由于组合逻辑在ASIC中由逻辑门和金属线构成而在 FPGA中由 LUT 和布线资源组成，因此ASIC 近似乘法器在 FPGA 上往往无法获得相同程度的硬件性能提升，因此本文面向 FPGA 架构提出了一种基于贝叶斯优化的自动化方法。最后，在实际应用中，通常需要选择库中的近似乘法器以使给定的大规模电路整体PPA最优，因此如何从逻辑综合的角度为一个大型设计挑选出使电路整体性能最高的近似乘法器单元是一个关键的问题。

- 评审意见3: 应在摘要、绪论的本文主要工作和组织结构及章引言中阐明各部分内容之间的逻辑关系，使得各部分内容串起来成为一个有机的整体。

修改说明：

感谢评审专家的意见。针对论文逻辑性差、拼凑感强这一问题，论文按照意见对摘要、绪论章的本文主要工作和组织结构、第3章到第5章的章引言进行了重新梳理和撰写，主要修改内容如下：

(1) 论文在摘要叙述各个工作具体内容之前，阐述了问题的研究背景及三个工作之间的联系，加强论文的逻辑性。

(2) 论文在摘要末尾，简要地总结了论文的主要工作及贡献，使摘要的行文逻辑更加清晰易懂。

(3) 论文对绪论章的本文主要工作和组织结构进行了重新梳理和改写，增加图1-3描述本文组织结构，更直观地阐明了本文各部分研究工作之间的关联性和系统性，并修改文字内容以描述三个主要工作之间的逻辑关系，使论文变成一个整体。

(4) 论文重新撰写了第3章到第5章的章引言部分，加强了各研究工作的逻辑及关联性描述，把论文串起来。第3章的引言描述了现有ASIC近似乘法器设计方法没有同时考虑数据分布和输入极性的缺点，引出第3章的研究内容。第4章的引言阐述了由于底层结构的不同，ASIC近似乘法器无法在FPGA上取得同等程度的性能提升，提出面向FPGA的自动化近似乘法器设计方法。第5章的引言讲述了基于第3章和第4章得到的近似乘法器，通过强化学习方法对不同近似乘法器的DNN硬件加速器进行近似逻辑综合的研究。

二、进一步地修改

(1) 论文改写了第2章的本章小节，描述已有研究工作的基础上，重点说明了存在的问题，为第3章到第5章的研究工作做准备。

(2) 对摘要、论文主体等出现的口语化内容、错别字和错误的标点符号进行修正。

(3) 对不合理的图片大小进行调整。如将图2-35、2-36（原图5-9）适当放大，将图2-32适当缩小（原图5-6）。

(4) 论文改写了第6章 总结与展望，在总结部分强化了各个工作之间的联系，并简化了各个工作的具体内容，提炼了每个研究工作的创新性结果并加以描述，提升文章的阅读体验。


最后，非常感谢您审阅我的论文，并提出了宝贵的意见，我收获颇丰，意识到了简单拼凑并不能成为一个合格的博士论文。您的建议很大程度地提高了我的论文写作水平和对论文系统性和关联性的理解，感谢您的指正！



#### Reference

[答辩公示管理应用](https://yzsfwapp.fudan.edu.cn/gsapp/sys/dbgsglappfudan/*default/index.do#/dbgs)

<!--- 邹鹏 王婧琦 -->

## 教训

1. 摘要要采取“总-分-总”的结构，第一次“总”介绍各工作之间的递进关系，第二次“总”总结全文，给出论文存在的意义；

2. 不能拼凑，不能每一章都有研究背景、研究现状、研究动机，这些内容的标题只应该出现在第一章，具体内容可以分散在第一章和第二章；

3. 画一张图来展示本文组织结构，该加的箭头要加，不该加的箭头不能加；（`很重要`）

4. 涉及到具体工作，每一章都应该有引言，由浅入深地阐述该章和前面章节的联系;

5. 每一章的小节以及最后的总结与展望，应着重强调逻辑性。