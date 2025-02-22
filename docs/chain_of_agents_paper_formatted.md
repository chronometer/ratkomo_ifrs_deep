# Chain of Agents: Large Language Models Collaborating on Long-Context Tasks

## Authors
Yusen Zhang, Ruoxi Sun, Yanfei Chen, Tomas Pfister, Rui Zhang, Sercan Arik  
*Penn State University, Google Cloud AI Research*  
{yfz5488, rmz5227}@psu.edu, {ruoxis, yanfeichen, tpfister, soarik}@google.com

## Abstract

Addressing the challenge of effectively processing long contexts has become a critical issue for Large Language Models (LLMs). Two common strategies have emerged:

1. Reducing the input length, such as retrieving relevant chunks by Retrieval-Augmented Generation (RAG)
2. Expanding the context window limit of LLMs

However, both strategies have drawbacks:
- Input reduction has no guarantee of covering the part with needed information
- Window extension struggles with focusing on the pertinent information for solving the task

To mitigate these limitations, we propose Chain-of-Agents (CoA), a novel framework that harnesses multi-agent collaboration through natural language to enable information aggregation and context reasoning across various LLMs over long-context tasks.

## Framework Overview

CoA consists of:
1. Multiple worker agents who sequentially communicate to handle different segmented portions of the text
2. A manager agent who synthesizes these contributions into a coherent final output

Key features:
- Processes the entire input by interleaving reading and reasoning
- Mitigates long context focus issues by assigning each agent a short context
- Demonstrates significant improvements by up to 10% over strong baselines of RAG, Full-Context, and multi-agent LLMs

## Architecture

### Stage 1: Worker Agent - Segment Comprehension and Chain-Communication
- Workers process different segments of the input sequentially
- Each worker communicates findings to the next worker in the chain
- Information is aggregated and refined through the chain

### Stage 2: Manager Agent - Information Integration and Response Generation
- Synthesizes contributions from all workers
- Generates coherent final output
- Ensures consistency and completeness

## Key Advantages

1. **Length Adaptability**
   - Extensible to inputs of different lengths
   - Adjustable number of worker agents
   - Last worker can read full input regardless of length

2. **Collaborative Communication**
   - Workers share relevant evidence
   - Progressive information building
   - Complex reasoning across segments

3. **Performance Improvements**
   - Outperforms baselines on various tasks:
     - Question answering
     - Summarization
     - Code completion

4. **Scalability**
   - Performance increases with longer inputs
   - Significant improvements over vanilla baselines
   - Effective handling of very long contexts

## Implementation Details

1. **Worker Agent Design**
   - Each worker processes a segment of input
   - Generates communication units for next worker
   - Maintains context through chain communication

2. **Manager Agent Design**
   - Integrates information from all workers
   - Resolves conflicts and inconsistencies
   - Produces final coherent output

3. **Communication Protocol**
   - Structured message passing between agents
   - Progressive information accumulation
   - Context preservation through chain

## Evaluation Results

1. **Task Performance**
   - Question Answering: Improved F1 scores
   - Summarization: Better ROUGE metrics
   - Code Completion: Higher code similarity scores

2. **Scaling Behavior**
   - Better performance with longer inputs
   - Consistent improvements across tasks
   - Robust to input length variations

3. **Ensemble Approaches**
   - Bi-directional processing
   - Self-consistency checks
   - Permutation-based improvements

## Technical Considerations

1. **Time Complexity**
   - Reduced from O(n²) to O(nk)
   - n = number of input tokens
   - k = context limit of LLM

2. **Implementation Challenges**
   - Catastrophic collapse prevention
   - Long-distance dependency handling
   - Context window optimization

## Future Directions

1. **Model Improvements**
   - Enhanced worker coordination
   - Better information synthesis
   - More efficient communication protocols

2. **Application Extensions**
   - New task domains
   - Different input types
   - Specialized agent roles

## Conclusion

Chain-of-Agents (CoA) presents a novel approach to handling long-context tasks through multi-agent collaboration. Its key innovations in sequential processing and information synthesis demonstrate significant improvements over existing methods, while maintaining scalability and adaptability across different tasks and input lengths.
