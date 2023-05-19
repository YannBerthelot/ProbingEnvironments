# ProbingEnvironments
ProbingEnvironments is a library that provides Reinforcement Learning Environments allowing for easy debugging of DeepRL actor-critic algorithms. Tired of debugging your agent by running it on CartPole or another Gym Env and not being sure if it works or you have bugs that cancel one another? This library aims at providing testing envs to make sure that each individual part of your actor-critic algorithm works on simple cases, this allows you to narrow down your bug chase.

The goal of this library is either :
- To use the environments yourself to check your agent by hand
- To include the premade tests in your units tests, allowing to check your agent without relying on long training tests on more complex environments

Functionnalities :
- Simple environments (in the gym framework) allowing to identify the part of your actor-critic algorithm that seems to be faulty.
- Premade tests/checks that wraps the enviroments and your agent to easily use those environments by hand or in your unit tests.
- Premade connectors to connect your agent to the tests (to adapt to the way you coded your agent without requiring refactoring) and a template to create yours.


# Disclaimer
The idea for this library comes from this presentation from Andy L Jones : https://andyljones.com/posts/rl-debugging.html



