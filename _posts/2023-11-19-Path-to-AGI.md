---
layout: post
title: "Exploring the Path to Artificial General Intelligence: Insights from Google DeepMind's Framework"
---

<div class="img-div-any-width" markdown="0" style="width: 50%; margin: 0 auto;">
  <img src="https://raw.githubusercontent.com/marinone94/marinone94.github.io/master/assets/img/path_to_agi/main_img.png"/>
<br />
</div>

Artificial General Intelligence (**AGI**) has long been a concept bordering on science fiction. However, with rapid advancements in Machine Learning and AI, what was once a subject of philosophical debate is now a matter of practical significance. **Google DeepMind**, probably one of the top 3 AI labs worldwide, has just introduced [a comprehensive framework for classifying AGI models](https://arxiv.org/abs/2311.02462), analogous to the levels of autonomous driving. This framework not only serves to measure progress toward AGI but also provides a common language for comparing models and assessing risks.

The concept of AGI is complex, often leading to various interpretations among experts. AGI is generally used to describe an AI system that can perform any intellectual task that a human being can. However, as DeepMind's paper illustrates, ask a hundred AI experts to define AGI, and you'll likely receive a hundred different definitions. This ambiguity necessitates a standard framework to objectively assess and understand the progress in AGI development.

In this article, I try to sum up the most important parts of the paper. It will be pretty much "bullet-pointish", but it feels like the best way to convey the message. I will also add my two cents at the end of the article. I hope you will enjoy it.

If you just want to read the Six Principles introduced, you can jump to **[DeepMind's AGI Framework: The Six Foundational Principles](#deepminds-agi-framework-the-six-foundational-principles)**. You can save some more time by reading only the bolded **For Dummies** highlights.
If you decide to do so, well, I hope you will give yourself the time to read the whole article and the paper shortly. It's worth it.

# Table of Contents
1. [A touch of history](#a-touch-of-history)
2. [DeepMind's AGI Framework: The Six Foundational Principles](#deepminds-agi-framework-the-six-foundational-principles)
3. [The Levels of AGI: A Nuanced Categorization](#the-levels-of-agi-a-nuanced-categorization)
4. [Generality Spectrum](#generality-spectrum)
5. [Testing for AGI](#testing-for-agi)
6. [Risk in Context: Autonomy and Human-AI Interaction](#risk-in-context-autonomy-and-human-ai-interaction)
7. [My (worthless) two cents](#my-worthless-two-cents)
8. [Epilogue](#epilogue)

# The paper


## A touch of history
The journey towards comprehending Artificial General Intelligence (AGI) is paved with diverse definitions and conceptualizations. Google DeepMind's paper presents nine case studies, each offering a unique perspective on AGI:

1. **The Turing Test:** Initially proposed by Alan Turing in 1950, this test evaluates if a machine can convincingly imitate human behavior. While the Turing Test has been passed by modern Large Language Models (LLMs), it's critiqued for highlighting the ease of deceiving humans rather than assessing true machine intelligence. Thus, the paper argues for a focus on capabilities rather than processes in defining AGI.

2. **Strong AI – Systems Possessing Consciousness:** Philosopher John Searle's concept of strong AI considers computers as not just tools but entities with minds and cognitive states. However, the lack of consensus on how to measure consciousness in machines makes this approach impractical for AGI.

3. **Analogies to the Human Brain:** Mark Gubrud's 1997 definition of AGI involves AI systems rivaling or surpassing human brain complexity. While modern machine learning systems draw inspiration from neural networks, the success of non-brain-like architectures like transformers suggests that brain-based benchmarks are not essential for AGI.

4. **Human-Level Performance on Cognitive Tasks:** Proposed by Legg and Goertzel, this definition views AGI as machines capable of performing tasks typically done by humans. This definition, however, leaves open questions about the range of tasks and the benchmark for human performance.

5. **Ability to Learn Tasks:** Murray Shanahan's definition in "The Technological Singularity" posits AGI as AI that can learn a broad range of tasks like a human, emphasizing the inclusion of metacognitive tasks (like learning) as a requirement for AGI.

6. **Economically Valuable Work:** OpenAI's charter defines AGI as systems outperforming humans in most economically valuable work. While focusing on performance regardless of underlying mechanisms, this definition might not capture the full spectrum of intelligence, such as artistic creativity or emotional intelligence.

7. **Flexible and General – The "Coffee Test" and Related Challenges:** Marcus’ view of AGI emphasizes flexibility, resourcefulness, and reliability, akin to human intelligence. He proposes concrete tasks like cooking in a kitchen or writing a bug-free program as benchmarks, although he acknowledges the need for a more comprehensive set of tasks for assessing AGI.

8. **Artificial Capable Intelligence (ACI):** Proposed by Mustafa Suleyman, ACI refers to AI systems capable of performing complex, multi-step tasks in the real world. Suleyman's “Modern Turing Test” involves an AI turning a certain amount of capital into a larger sum, highlighting the emphasis on complex real-world tasks.

9. **State-of-the-Art LLMs as Generalists:** Agüera y Arcas and Norvig argue that current LLMs like GPT-4 already qualify as AGIs due to their generality – their ability to handle a wide range of topics, tasks, and languages. However, the paper suggests that this generality must be paired with reliable performance to truly classify as AGI.

Each case study gives us a different way to look at AGI, helping us get a better and more detailed picture of this complicated area. The paper uses these studies to support its idea of creating different levels for understanding AGI, mixing the complex nature of human intelligence with the real-world abilities of AI.

## DeepMind's AGI Framework: The Six Foundational Principles
DeepMind's framework is anchored on six principles, each contributing a critical dimension to the understanding of AGI:

1. **Capabilities Over Processes:**
Focusing on the achievements of AGI rather than the underlying processes, this principle advocates assessing AGI based on its outcomes. It stresses the importance of the results over the means employed to attain them. **For Dummies: It's about what AGI can achieve, not how it gets there.**

2. **Separating Generality and Performance:**
This principle distinguishes between the breadth (generality) and depth (performance) of AGI's capabilities. It emphasizes the need to assess AGI not just on its proficiency in specific tasks (performance) but also on its adaptability and versatility across a range of tasks (generality). **For Dummies: It's about how well AGI can perform a task and how many tasks it can perform.**

3. **Emphasis on Cognitive Tasks:**
Prioritizing tasks that demand mental prowess over physical abilities, this principle underscores the cognitive aspect of AGI. It is more concerned with AGI's ability to reason, learn, and understand, rather than its physical operational capabilities. **For Dummies: It's about how well AGI can think, not how well it can move.**

4. **Potential vs. Deployment:**
Highlighting a forward-looking perspective, this principle differentiates between what AGI is currently capable of (deployment) and what it might achieve in the future (potential). It encourages an outlook that considers the future possibilities and evolution of AGI. **For Dummies: It's about what AGI can do now and what it can do in the future.**

5. **Real-World Benchmarking:**
This principle argues for the necessity of testing AGI in real-world scenarios. It advocates for the ecological validity of AGI testing, ensuring that AGI systems can navigate and adapt to the complexities and unpredictability of real-life situations.
**For Dummies: It's about how well AGI can perform in the real world, not in a lab.**

6. **Journey Before Destination:**
Emphasizing the evolutionary path of AGI, this principle focuses on the progression towards achieving AGI, rather than fixating on the final goal. It recognizes the incremental developments and milestones as critical components of the AGI journey.
**For Dummies: It's about the path to AGI, not just the destination.**

## The Levels of AGI: A Nuanced Categorization

DeepMind’s framework also introduces a nuanced categorization of AGI into levels based on performance and generality, providing a more granular understanding of AGI's developmental stages:

0. **No AI:**
Representing the absence of AI capabilities, this level serves as the baseline against which AI development is measured.

1. **Emerging:**
At this initial stage, AGI systems exhibit basic capabilities, slightly outperforming an unskilled human in certain tasks. It marks the beginning of AI's functional utility.

2. **Competent:**
AGI systems at this stage match the average skill level of a human adult. They demonstrate competence in a range of tasks, indicating a significant advancement in AI capabilities.

3. **Expert:**
This advanced stage of AGI represents systems that exceed the capabilities of most humans, achieving top-tier performance in several areas.

4. **Virtuoso:**
The pinnacle of AGI development, where AI capabilities surpass those of nearly all humans. AGI systems at this level exhibit exceptional skill and proficiency.

5. **Superhuman:**
It simply outperforms 100% of humans (e.g.: AlphaFold in protein folding is a Narrow ASI). General ASIs might also be able to perform cognitive tasks that no human is capable of (e.g.: to speak with animals).

## Generality Spectrum
The framework also introduces a Generality spectrum, ranging from 'Narrow' (specialized in specific tasks) to 'General' (capable across a wide range of tasks). This spectrum provides a measure of AGI's versatility and adaptability.

The combination of these levels of performance and generality dimensions provides a comprehensive framework to classify AGI systems. It allows for a nuanced understanding of where a particular AI system stands in terms of its capabilities and how close it is to achieving a more generalized intelligence comparable to human capabilities.

## Testing for AGI
The paper then addresses the challenges in benchmarking AGI. Traditional AI benchmarks, often task-specific and narrow, fall short in evaluating AGI's broader capabilities.

The authors discuss how their six proposed principles for defining AGI inform the creation of a matrixed leveled ontology to assess AI capabilities. They emphasize the need for a benchmark that includes a broad suite of cognitive and metacognitive tasks, encompassing diverse aspects like linguistic intelligence, mathematical and logical reasoning, spatial reasoning, social intelligence, learning new skills, and creativity.

The chapter also raises questions about the set of tasks that constitute the generality criteria for AGI and the proportion of these tasks an AI system must master to achieve a certain level of generality. It suggests that the benchmark should include open-ended and interactive tasks, which, despite being challenging to measure, offer better ecological validity than traditional AI metrics. The authors propose that an AGI benchmark should be a living benchmark, adaptable to include new tasks and evolving standards, despite the inherent imperfections in measuring such complex concepts.

They also raise an open question: when benchmarking performance, should AIs be allowed to use tools, including potentially AI-powered tools? Most likely, this choice should be task-dependent. 

One final note: systems that pass the large MAJORITY of the tasks associated with a certain AGI level should be considered to have reached such a level.

## Risk in Context: Autonomy and Human-AI Interaction

Last, the paper highlights the importance of considering risk, including existential risks ("x-risk"), in the context of AGI development. It emphasizes the criticality of carefully chosen human-AI interaction paradigms for the safe and responsible deployment of advanced AI models. The authors underline the significance of sometimes opting for a "No AI" paradigm, particularly in situations where AI's involvement may not be desirable or safe, such as in certain educational, recreational, assessment, or safety-sensitive contexts.

Furthermore, the section discusses how the principles and ontology proposed in the document can reshape the conversation around the risks associated with AGI. It's noted that AGI and autonomy are not synonymous; the document introduces "Levels of Autonomy" that are enabled by but not determined by AGI's progression. This distinction allows for a more nuanced understanding of the risks associated with AI systems and underscores the necessity of concurrent advancements in human-AI interaction research and AI model development​. You should refer to the paper for a detailed presentation of the Levels of Autonomy, as this article is - once again - way too long.

# My (worthless) two cents
The paper raises important considerations regarding the capabilities and classification of AGI systems. One key point of discussion is whether the ability to control physical actuators should be a criterion for AGI. The paper suggests that while physical capabilities, such as the ability to manipulate objects or perform tasks in the physical world, can enhance the generality of an AI system, they are not deemed essential prerequisites for achieving AGI​​.

In light of this, I would argue that a comprehensive framework for AGI should not dismiss the importance of physical interaction capabilities. The ability to interact with and manipulate the physical environment is a crucial aspect of intelligence. We distinguish between humans and other animals based on our ability to control the physical environment at least as much as our cognitive skills. Thus, I would suggest that the framework should include a category for physical interaction capabilities, which can be further divided into fine motor skills (e.g., manipulating objects) and gross motor skills (e.g., locomotion).

Additionally, the discussion on the generality spectrum in the DeepMind paper seems somewhat limited. To enhance this spectrum, I propose several categories for consideration:

1. **Single Task vs Multi-Task Capability:** Evaluating whether an AI system is specialized in a single task or capable of handling multiple tasks.

2. **Zero-Shot Learning vs Re-training:** Assessing whether the AI can adapt to new tasks without prior training or if it requires re-training for each or some of the new tasks it encounters.

3. **Single Modality vs Multimodal I/O:** Considering if the AI can process and respond to only one type of input and produce only one kind of output or if it can handle multiple modalities (e.g., visual, auditory, textual).

4. **Digital vs Physical Output:** Distinguishing between AI systems that only interact in digital formats and those capable of producing physical outputs.

5. **Single vs Multiple Agents:** Evaluating whether the AI system is a single agent or a multi-agent system.

5. **LAST, AND MOST IMPORTANT, Models vs Systems:** Comparing pears with pears and apples with apples (a.k.a comparing Models with Models and Systems with Systems). We should once and for all agree to distinguish the Model evaluation and the System evaluation. Comparing Systems (usually accessible via proprietary APIs whose underlying code is not publicly disclosed) with standalone models is like comparing pears with apples, and we are taught in primary school that's not the best way to do science. And we see it happening way too often.

Incorporating these categories would provide an even more nuanced and comprehensive understanding of AGI's capabilities, aligning the framework with the multifaceted nature of intelligence and its applications. Happy to discuss this further with anyone interested. Just reach out!

# Epilogue
In this era of rapid AI advancements, frameworks like DeepMind's are not just useful; they're essential. They guide researchers, developers, and policymakers through the complex landscape of AGI, ensuring that as we march towards this new frontier, we do so with our eyes wide open to both the opportunities and the challenges that lie ahead. What an exciting time to be alive!