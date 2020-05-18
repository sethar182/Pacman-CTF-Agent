Hello, thanks for your interest in my code!

This goal of this project was to create a team of pacman agents that could
compete in a capture-the-flag version of pacman. The rules and restrictions
of the competition can be found in more detail in contest.html.

To run the program, I use python in the command line. A valid example is:
```
py capture.py -r baselineTeam -b step_pac
py capture.py -r myTeam_OLD -b step_pac
py capture.py -r step_pac -b step_pac
```
The program uses a minimax adversarial search with alpha-beta pruning. At each
leaf of the tree, my team's agents are evaluated based on a linear combination
of extracted features and weights. My team has at most one defending agent, and
the rest will be attacking agents. 

The biggest challenge of this project was finding appropriate weights to use
in the evaluation function. With more time, I would like to implement a machine
learning approach to finding optimal weights. My final team is step_pac.py.