## Pretraining

---
A Map for Studying Pre-training in LLMs
- Data Collection
   - General Text Data
   - Specialized Data
- Data Preprocessing 
   - Quality Filtering
   - Deduplication
   - Tokenization
- How Does Pretraining Affect LLMs?
   - Mixture of Sources
   - Amount of Pretraining Data
   - Quality of Pretraining Data
- Architecture for Pretraining
   - Encoder Decoder Architecture 
   - Causal Decoder Architecture 
   - Prefix Decoder Architecture 
   - Emergent Architectures
- Some Notes on Configurations 
   - Layer Normalization
   - Attention
   - Positional Encoding
- Pretraining Tasks 
   - Language Modeling
   - Denoising Autoencoding
   - Mixture-of-Denoisers
- ~~Decoding Strategy~~
- Why Does Predicting the Next Word Work?
- ~~Model Training~~

---
Pre-training is the foundation for the power of Large Language Models (LLMs) like me. By training on lots of text data, these models get good at understanding and generating language.
- **Importance of Pre-training Data:**
    * LLMs need to be trained on a lot of data to become really capable.
    * The size and quality of this data are super important. Good data helps the model achieve better capabilities.
- **Structure of the Discussion:**
    * Section 4.1 will talk about how data is collected and processed.
    * Section 4.2 will discuss the popular designs or blueprints (known as architectures) used for these models.
    * Section 4.3 will describe methods to train LLMs efficiently and without issues.
In simpler terms, before LLMs like me can be useful, they need to be trained on lots of good quality data. How this data is gathered, the design of the model, and the methods to train it are all vital components of the process.


### Data collection 

Since the goal of an llm is ambitious in nature, the data needed for pretraining should be high quality and voluminous.


There are two main types of data used to train Large Language Models (LLMs): general data and specialized data.
* General Data:
    * Examples: Web pages, books, chat logs.
    * Why it's used: It's widely available, varied, and helps LLMs get good at general language understanding and adaptability.
* Specialized Data:
    * Examples: Datasets in multiple languages, scientific content, and programming code.
    * Purpose: Helps LLMs become experts in specific areas or tasks.

while LLMs are often trained on general data to understand everyday language, it was found that they can also be trained on specific types of data to get better at specialized tasks.

#### General Text Data: 
Most Large Language Models (LLMs) use general-purpose data. Let's look at three key types:
- Webpages:
    * What it offers: A wide range of data from the internet that provides diverse language knowledge.
    * Example Dataset: CommonCrawl.
    * Issues: Web data can have both high-quality text (like Wikipedia) and low-quality text (like spam). It's crucial to clean and process this data to ensure quality.
- Conversation Text:
    * Why it's useful: Helps LLMs get better at conversation and question-answering.
    * Example Dataset: PushShift.io Reddit corpus.
    * How it's used: Conversations from online platforms are structured into tree-like formats to capture responses. This allows multi-party chats to be split into smaller conversations for training.
    * Challenges: Relying too much on dialogue data can lead to problems. For instance, LLMs might misunderstand certain phrases as conversation starters, affecting their response quality.
- Books:
    * Why they matter: Books offer formal and lengthy texts which help LLMs understand complex language structures, grasp long-term context, and produce coherent narratives.
    * Example Datasets: Books3 and Bookcorpus2 found in the Pile dataset.
In essence, these general data sources help LLMs understand and generate varied and natural language, but each source comes with its unique strengths and challenges.


#### Specialized Text Data: 
Specialized data helps LLMs get better at certain specific tasks. Here are three types of specialized data:
- Multilingual Text:
    * Purpose: Improving language understanding and generation across multiple languages.
    * Example Datasets: BLOOM (covers 46 languages) and PaLM (covers 122 languages).
    * Benefit: These models are great at tasks like translation, multilingual summaries, and multilingual Q&A, sometimes even outperforming models trained just on target language data.
- Scientific Text:
    * Why it's used: Helps LLMs understand scientific knowledge.
    * Source: Materials like arXiv papers, scientific textbooks, math websites, and more.
    * Challenges & Solutions: Scientific texts have things like math symbols and protein sequences. To handle this, they're specially tokenized and pre-processed to fit a format LLMs can use.
    * Benefits: LLMs trained on scientific texts are better at scientific tasks and reasoning.
- Code:
    * Why it's relevant: Training LLMs on code helps with program synthesis, a popular research area.
    * Current State: Even powerful LLMs, like GPT-J, find it tough to produce good, accurate programs.
    * Source: Code can come from Q&A communities like Stack Exchange or public software repositories like GitHub. This includes actual code, comments, and documentation.
    * Challenges & Solutions: Code has its own syntax and logic, and is very different from regular text. Training on code, though, might give LLMs complex reasoning skills.
    * Benefits: When tasks are formatted like code, LLMs can produce more accurate answers.
In short, using specialized data gives LLMs specific skills, from understanding multiple languages to generating code.

### Data Preprocessing 
Data preprocessing can be broadly broken down into the following three categories:

* Quality Filtering
* Deduplication
* Tokenization


| Preprocessing Technique | Description |
| --- | --- |
| Quality Filtering | Removing low-quality data from the corpus using either classifier-based or heuristic-based approaches. Classifier-based methods train a binary classifier with well-curated data as positive instances and sample candidate data as negative instances, and predict the score that measures the quality of each data example. Heuristic-based methods use rules or heuristics to filter out low-quality data based on certain criteria. |
| Deduplication | Duplicate documents can arise from various sources, such as web scraping, data merging, or data augmentation, and can lead to several issues, such as overfitting, bias, or inefficiency. To address these issues, existing studies mostly rely on the overlap ratio of surface features (e.g., words and n-grams overlap) between documents to detect and remove duplicate documents containing similar contents. Furthermore, to avoid the dataset contamination problem, it is also crucial to prevent the overlap between the training and evaluation sets, by removing the possible duplicate texts from the training set. It has been shown that the three levels of deduplication (i.e., document-level, sentence-level, and token-level) are useful to improve the training of LLMs, which should be jointly used in practice.  |
| Tokenization | Splitting the raw text into individual tokens or subword units, which can be fed into the model as input. This can be done using various algorithms, such as whitespace-based, rule-based, or statistical methods. |

```
Note: The overlap ratio is the percentage of words or n-grams that two documents have in common. 

It is calculated by dividing the number of shared words or n-grams 
by the total number of words or n-grams in the two documents.

For example, if two documents have the following sentences:

Document 1: "I love to eat pizza."
Document 2: "I love to eat pizza with my friends."

Then the overlap ratio of the two documents would be 50%, 
because they share two words ("love" and "pizza") in common.

The overlap ratio can be used to identify duplicate documents. 
If two documents have a high overlap ratio, then they are likely to be duplicates.

The overlap ratio can also be used to measure the similarity between two documents. 
If two documents have a low overlap ratio, then they are likely to be very different.

Overlap ratio is commonly used in natural language processing (NLP) tasks, 
such as deduplication, text classification, and machine translation.

Imagine you have two baskets of fruit. If the two baskets contain the same fruits, 
then the overlap ratio is 100%. If the two baskets contain no fruits in common, 
then the overlap ratio is 0%. If the two baskets contain some fruits in common, 
but not all of them, then the overlap ratio is somewhere between 0% and 100%.

The overlap ratio can be used to determine how similar the two baskets of fruit are. 
If the overlap ratio is high, then the two baskets are likely to be very similar. 
If the overlap ratio is low, then the two baskets are likely to be very different.
```

### How does pretraining affect LLMs?

Effect of pretraining on LLMs can be broadly categorized into three categories
* Mixture of sources
* Amount of pretraining data
* Quality of pretraining data

- Imagine you are preparing for an exam that costs you thousands of dollars in exam fees but also a lot of time and information to prepare. It is needless to say that you will not be enthusisastic about retaking the exam again and again.
- Training an LLM is somewhat similar. The cost incurred both in terms of resources and architecture is so massive that it is not a recommended practice to train an LLM over and over for every new task. Thus the pretraining of an LLM must equip it with  a good arsenal of parameters so that it can generalize seamlessly to downstream tasks later.

| **Topic**                  | **Key Points**                                                                                                                                                                                                                                                                                 | **Practical Implications**                                                                                                                                |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Mixing Sources             | - Combining text data from different areas can give LLMs a wider range of knowledge and improve their ability to generalize across different tasks. <br> - The key is to include many high-quality data sources, and the way data from different sources is mixed (its distribution) is crucial. <br> - Researchers should think deeply about how much data from each source they use in pre-training, and the goal is to develop LLMs that fit their specific needs without sacrificing general capabilities. | - Consider the diversity of data sources when pre-training LLMs. <br> - Experiment with different data distributions to find what works best for your specific needs. <br> - Be mindful of the potential pitfalls of over-relying on data from one domain.                  |
| Amount of Pre-training Data | - High-quality data in large quantities is crucial for effective pre-training of LLMs. <br> - There's a strong correlation between the size of the LLM and the required data size for optimal training. <br> - Many LLMs don't reach their full potential due to insufficient pre-training data. <br> - There's evidence to suggest that increasing both the model and data size proportionally leads to more compute-efficient models. <br> - Smaller models can yield impressive results when given more data and longer training durations.                                         | - When scaling up model parameters, consider the adequacy of your training data. <br> - Focus on both the quantity and quality of the data. <br> - Experiment with different model sizes and data amounts to find the sweet spot for your specific needs.                        |
| Quality of Pre-training Data| - The quality of the data on which LLMs are trained significantly impacts their performance. <br> - Pre-training on low-quality data can be detrimental to model performance. <br> - Studies have shown that models trained on high-quality, cleaned data perform better on downstream tasks. <br> - Data duplication can introduce a series of issues, such as double descent, excessive duplication dominating the training process, and degraded copy from context capability. <br> - Applying meticulous pre-processing on the pre-training corpus is essential to ensure stability during training and prevent unintended negative impacts on model performance.         | - Curate high-quality datasets for pre-training LLMs. <br> - Clean and filter data of noise, toxicity, and duplications. <br> - Experiment with different pre-processing techniques to find what works best for your specific dataset.                                        |



### Pretraining Tasks
There are two commonly used pretraining tasks:

1. Language Modelling
2. Denoising Autoencoding


**Language Modeling in LLMs**:

Language Modeling (LM) is a foundational task in natural language processing. It entails predicting the next token in a sequence based on its history. When applied to Large Language Models (LLMs), especially decoder-only models, it serves as a pivotal pre-training task. Here's a concise summary of the provided content:

1. **Definition**:
   - Language Modeling, as a task, seeks to predict a token $$\( xi \)$$ in a sequence based on all the preceding tokens $$\( x \lt i \)$$.
   - Given a token sequence $$\( x = {x1,...,xn} \)$$, the goal is to autoregressively predict each token.
   - The objective can be mathematically defined as:
     $$\[ LLM(x) = \sum_{i=1}^{n} log P(xi|x \lt i) \]$$
   This expression essentially calculates the log likelihood of predicting each token $$\( xi \)$$ given the prior tokens in the sequence.

2. **Advantages for Decoder-only LLMs**:
   - Decoder-only models, like GPT3 [55] and PaLM [56], heavily utilize the LM task for pre-training.
   - A strong suit of these models is that many language tasks can be reshaped as prediction problems, which aligns seamlessly with the natural inclination of LLMs trained using an LM objective.
   - An intriguing observation is that some decoder-only LLMs can be applied to certain tasks by merely predicting the subsequent tokens autoregressively. This means they can sometimes perform tasks without the need for explicit fine-tuning [26, 55].

3. **Prefix Language Modeling Variant**:
   - A notable variation of the standard LM task is the prefix language modeling.
   - In this variation, only the tokens beyond a randomly selected prefix are considered for loss computation.
   - Despite the model seeing the same amount of tokens during pre-training as in standard LM, prefix language modeling typically underperforms standard LM because fewer tokens in the sequence are harnessed during model pre-training [29].

In essence, Language Modeling serves as the backbone of decoder-only LLM pre-training. Its autoregressive nature enables LLMs to implicitly learn a plethora of tasks, often without the necessity for task-specific fine-tuning. Adjustments and variations, like prefix language modeling, offer different ways to employ the task, but the foundational principle of predicting token sequences remains consistent.


**Denoising Tasks in LLM Pre-training**:

The discussion revolves around denoising tasks, which are prominent pre-training objectives for Large Language Models (LLMs). Let's break down the content:

1. **Denoising Autoencoding (DAE)**:
   - **Definition**: In the Denoising Autoencoding task, parts of the input text are intentionally corrupted by replacing certain spans. The objective is to train the model to recover the original, uncorrupted tokens.
   - **Formulation**: The task's training objective is represented as $$\( L_{DAE}(x) = log P (\tilde{x}|x\backslash \tilde{x}) \)$$. Here, $$\( \tilde{x} \)$$ refers to the replaced tokens, and the model is trained to predict these based on the corrupted input $$\( x\backslash \tilde{x} \)$$.
   - **Adoption**: While conceptually powerful, the DAE task can be more intricate to implement compared to the standard LM task. As such, it hasn't been as broadly adopted for LLM pre-training. However, models like T5 [73] and GLM-130B [84] use DAE as a pre-training objective and work to recover the replaced spans in an autoregressive manner.

2. **Mixture-of-Denoisers (MoD)**:
   - **Concept**: MoD, also known as the UL2 loss, offers a unified pre-training objective for language models. It posits that both the LM and DAE tasks can be treated as distinct forms of denoising tasks. 
   - **Types of Denoisers**:
     - **S-denoiser (LM)**: This is akin to the conventional Language Modeling objective.
     - **R-denoiser (DAE, short span and low corruption)**: A variant of DAE where short spans of text are corrupted.
     - **X-denoiser (DAE, long span or high corruption)**: Another DAE variant but with either longer corrupted spans or a higher corruption ratio.
   - **Usage**: Depending on the initial special tokens in input sentences (like {[R], [S], [X]}), different denoisers are used for model optimization. For instance, a sentence beginning with the token [S] would utilize the S-denoiser (LM).
   - **Applications**: MoD has been integrated into models like PaLM 2 [107].

**In Summary**: The essence of denoising tasks in LLM pre-training is to teach the model to recover corrupted or missing parts of input sequences. While the standard Language Modeling task remains dominant, denoising objectives like DAE and MoD provide alternative methods to pre-train and refine the capabilities of Large Language Models.


### Why Does Predicting the next word work?

> Say you read a detective novel. It’s
> like complicated plot, a storyline,
> different characters, lots of events,
> mysteries like clues, it’s unclear.
> Then, let’s say that at the last
> page of the book, the detective has
> gathered all the clues, gathered
> all the people and saying, "okay,
> I’m going to reveal the identity of
> whoever committed the crime and that
> person’s name is". Predict that word.
> ...
> Now, there are many different words.
> But predicting those words better and
> better, the understanding of the text
> keeps on increasing. GPT-4 predicts
> the next word better
>
> - Ilya Sutskever


**Influence of Architecture and Pre-training Tasks on LLMs**:

1. **Architecture Choice**:
   - **Discussion**: Early literature on pre-trained language models extensively discussed architectural effects. However, many LLMs use the causal decoder architecture, with limited theoretical analysis on its advantages.
   - **Causal Decoder and LM Objective**: 
     - LLMs using a causal decoder architecture with a language modeling (LM) objective have shown strong zero-shot and few-shot generalization capabilities.
     - Without multi-task fine-tuning, the causal decoder performs better in zero-shot scenarios than other architectures.
     - GPT-3, a popular model, confirmed that large causal decoders can be effective few-shot learners.
     - Techniques like instruction tuning and alignment tuning can enhance the performance of large causal decoder models.
   - **Scaling Law**:
     - Causal decoders benefit from scaling laws: increasing model size, dataset size, and computation can notably improve performance.
     - In-depth studies on encoder-decoder models, especially at larger scales, are needed.
   - **Future Research**:
     - More research is necessary to understand how architecture and pre-training task choices affect LLM capacity. Particular interest is in encoder-decoder architectures. Additionally, detailed LLM configurations deserve more attention.

2. **Long Context**:
   - **Context Limitation**: Transformers have traditionally been constrained by context length due to quadratic computational costs in terms of time and memory.
   - **Growing Demand**: With increasing needs for long context windows in tasks like PDF processing and story writing, models are evolving. For instance, ChatGPT has expanded its context window from 4K tokens to 16K tokens, and GPT-4 has been extended to 32K tokens.
   - **Extrapolation**:
     - This refers to an LLM's ability to handle input texts that are longer than the maximum length seen during training.
     - Position embedding techniques like RoPE and T5 bias have displayed extrapolation abilities. For example, LMs equipped with ALiBi have demonstrated consistent performance on sequences much longer than training sequences. Furthermore, the xPos method seeks to enhance the extrapolation capability of RoPE.
   - **Efficiency**:
     - To address the quadratic computational challenge, various studies have proposed more efficient attention computation methods, such as sparse or linear attentions.
     - FlashAttention improves efficiency at the system level (focusing on GPU memory IO efficiency), enabling training LLMs with longer context windows with the same computational budget.
     - Some researchers are proposing novel architectures to tackle this efficiency challenge, such as RWKV and RetNet.

**In Summary**: The architecture and pre-training tasks play a pivotal role in determining the capabilities and biases of Large Language Models. Current trends show a strong inclination towards causal decoder architectures, though more research is needed on alternative models, especially encoder-decoder architectures. Moreover, as applications demand more extensive context windows, LLMs are evolving, and innovations are emerging in both their ability to extrapolate and in their computational efficiency.
