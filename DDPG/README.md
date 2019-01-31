# introduction
DDPG (deep deterministic policy gradient) is a actor-critic algorithm for continuous control problem. DDPG contains two parts: actor based on q-learning and critic based on policy gradient, In every part, there are two neural networks with similar form of DQN algorithm. Evaluation network is trained on-line, and target network won't be trained, it just updated with parameters in evaluated network periodically. In conclusion, there are four neural networks in total. actor-evaluation (to be trained on-line), actor-target (with parameters of actor-evaluation), critic-evaluation (to be trained on-line), critic-target (with parameters of actor-evaluation). The uses of these four networks respectively are: learn q value, accelerating learning, learn action gradients, accelerating learning. 

# algorithm
![image](https://github.com/yw825137911/RL-tensorflow/blob/master/img/algorithm.png)

# resource
http://arxiv.org/abs/1509.02971

https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-2-A-DDPG/

https://github.com/floodsung/DDPG

https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
