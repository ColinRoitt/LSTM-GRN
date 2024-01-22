# LSTM-GRN
Implementation of LSTM-GRN in paper Enough is Enough: Learning to Stop in Generative Systems

# Experiments
Implemented are two experiments. A single line growing towards a single target, and a more complex branching system. See the paper for details.

## Implementation
LSTM GRN is implemented in the file `ELSTM.py` and is self contained. This code is based of Colah's blog post on [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). This is then extended into a static network that can be evolved over using GA. 

## Diagram
<img style="background: white; padding: 10px" src="https://raw.githubusercontent.com/ColinRoitt/LSTM-GRN/main/LSTM-GRN%20Diagram.svg">

